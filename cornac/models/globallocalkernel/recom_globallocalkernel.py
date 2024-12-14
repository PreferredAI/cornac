import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import logging


from ..recommender import Recommender

from tqdm import tqdm


# Configure logging
# logging.basicConfig(level=logging.INFO, format="%(message)s")
# ===========================
# Define your model layers
# ===========================

def local_kernel(u, v):
    dist = torch.norm(u - v, p=2, dim=2)
    hat = torch.clamp(1. - dist**2, min=0.)
    return hat

class KernelLayer(nn.Module):
    def __init__(self, n_in, n_hid, n_dim, lambda_s, lambda_2, activation=nn.Sigmoid()):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_in, n_hid))
        self.u = nn.Parameter(torch.randn(n_in, 1, n_dim))
        self.v = nn.Parameter(torch.randn(1, n_hid, n_dim))
        self.b = nn.Parameter(torch.randn(n_hid))

        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2

        # nn.init.xavier_uniform_(self.W, gain=torch.nn.init.calculate_gain("sigmoid"))
        # nn.init.xavier_uniform_(self.u, gain=torch.nn.init.calculate_gain("sigmoid"))
        # nn.init.xavier_uniform_(self.v, gain=torch.nn.init.calculate_gain("sigmoid"))

        nn.init.kaiming_uniform_(self.W,nonlinearity = "sigmoid")
        nn.init.kaiming_uniform_(self.u, nonlinearity = "sigmoid")
        nn.init.kaiming_uniform_(self.v,nonlinearity = "sigmoid")

        nn.init.zeros_(self.b)
        self.activation = activation

    def forward(self, x):
        w_hat = local_kernel(self.u, self.v)
        sparse_reg = torch.nn.functional.mse_loss(w_hat, torch.zeros_like(w_hat))
        sparse_reg_term = self.lambda_s * sparse_reg
        l2_reg = torch.nn.functional.mse_loss(self.W, torch.zeros_like(self.W))
        l2_reg_term = self.lambda_2 * l2_reg

        W_eff = self.W * w_hat  # Local kernelised weight matrix
        y = torch.matmul(x,  W_eff) + self.b
        y = self.activation(y)

        return y, sparse_reg_term + l2_reg_term

class KernelNet(nn.Module):
    def __init__(self, n_u, n_hid, n_dim, n_layers, lambda_s, lambda_2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:

                layers.append(KernelLayer(n_u, n_hid, n_dim, lambda_s, lambda_2))
            else:

                layers.append(KernelLayer(n_hid, n_hid, n_dim, lambda_s, lambda_2))
        # Output layer
        layers.append(KernelLayer(n_hid, n_u, n_dim, lambda_s, lambda_2, activation=nn.Identity()))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(0.33)

    def forward(self, x):
        total_reg = None
        for i, layer in enumerate(self.layers):
            x, reg = layer(x)
            if i < len(self.layers)-1:
                x = self.dropout(x)
            total_reg = reg if total_reg is None else total_reg + reg
        return x, total_reg

class CompleteNet(nn.Module):
    def __init__(self, kernel_net, n_u, n_m, n_hid, n_dim, n_layers, lambda_s, lambda_2, gk_size, dot_scale):
        super().__init__()
        self.gk_size = gk_size
        self.dot_scale = dot_scale
        self.local_kernel_net = kernel_net
        self.conv_kernel = torch.nn.Parameter(torch.randn(n_m, gk_size**2) * 0.1)
        nn.init.xavier_uniform_(self.conv_kernel, gain=torch.nn.init.calculate_gain("relu"))

    def forward(self, x, x_local):
        gk = self.global_kernel(x_local, self.gk_size, self.dot_scale)
        x = self.global_conv(x, gk)
        x, global_reg_loss = self.local_kernel_net(x)
        return x, global_reg_loss

    def global_kernel(self, input, gk_size, dot_scale):
        avg_pooling = torch.mean(input, dim=1)  # Item based average pooling (axis=1)
        avg_pooling = avg_pooling.view(1, -1)
        gk = torch.matmul(avg_pooling, self.conv_kernel) * dot_scale
        gk = gk.view(1, 1, gk_size, gk_size)
        return gk

    def global_conv(self, input, W):
        input = input.unsqueeze(0).unsqueeze(0)
        conv2d = nn.LeakyReLU()(F.conv2d(input, W, stride=1, padding=1))
        return conv2d.squeeze(0).squeeze(0)

class Loss(nn.Module):
    def forward(self, pred_p, reg_loss, train_m, train_r):
        diff = train_m * (train_r - pred_p)
        sqE = torch.nn.functional.mse_loss(diff, torch.zeros_like(diff))
        loss_p = sqE + reg_loss
        return loss_p

# ===========================
# Cornac Recommender Wrapper
# ===========================

class GlobalLocalKernel(Recommender):
    """Global-Local Kernel Recommender.

    Parameters
    ----------
    n_hid: int, default: 64
        Size of the hidden dimension.
    n_dim: int, default: 10
        Kernel dimension.
    n_layers: int, default: 2
        Number of kernel layers (not counting the final output layer).
    lambda_s: float, default: 0.001
        Sparsity regularization term.
    lambda_2: float, default: 0.001
        L2 regularization term.
    gk_size: int, default: 3
        Size of the global kernel.
    dot_scale: float, default: 0.1
        Scaling factor for the global kernel.
    max_epoch_p: int, default: 100
        Max epochs for pre-training phase.
    max_epoch_f: int, default: 100
        Max epochs for fine-tuning phase.
    tol_p: float, default: 1e-4
        Tolerance for early stopping in pre-training.
    tol_f: float, default: 1e-4
        Tolerance for early stopping in fine-tuning.
    patience_p: int, default: 10
        Patience for early stopping in pre-training.
    patience_f: int, default: 10
        Patience for early stopping in fine-tuning.
    lr: float, default: 0.001
        Learning rate.
    verbose: bool, default: False
        Whether to print training progress.
    device: str, default: 'auto'
        'cpu', 'cuda', or 'auto' to automatically detect GPU.
    """

    def __init__(
        self, 
        n_hid=1,       # size of hidden layers
        n_dim=2,         # inner AE embedding size
        n_layers=2,      # number of hidden layers


        # lambda_s=0.0001,  # regularization of sparsity of the final matrix
        # lambda_2=0.0001,    # regularization of number of parameters

        lambda_s=0.006,  # regularization of sparsity of the final matrix
        lambda_2=0.001,    # regularization of number of parameters

        gk_size=3,       # width=height of kernel for convolution
        dot_scale=1,     # dot product weight for global kernel
        max_epoch_p=10, # max number of epochs for pretraining
        max_epoch_f=10, # max number of epochs for finetuning
        tol_p=1e-4,      # min threshold for difference between consecutive values of train rmse for pretraining
        tol_f=1e-5,      # min threshold for difference for finetuning
        patience_p=10,    # early stopping patience for pretraining
        patience_f=10,   # early stopping patience for finetuning
        lr_p=0.01,
        lr_f=0.001, 
        verbose=False, 
        name="GlobalLocalKernel", 
        trainable=True
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.n_hid = n_hid
        self.n_dim = n_dim
        self.n_layers = n_layers
        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2
        self.gk_size = gk_size
        self.dot_scale = dot_scale
        self.max_epoch_p = max_epoch_p
        self.max_epoch_f = max_epoch_f
        self.tol_p = tol_p
        self.tol_f = tol_f
        self.patience_p = patience_p
        self.patience_f = patience_f
        self.lr_p = lr_p
        self.lr_f = lr_f
        self.verbose = verbose

        # Device
        if torch.cuda.is_available() and (self.device != 'cpu'):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.verbose = verbose
        self.model = None
        self.train_r_local = None

    def fit(self, train_set, val_set=None):
        # Prepare training data
        # train_r = train_set.csr_matrix.toarray().astype(np.float32)
        # print('train : ', train_r.shape) # (943, 1656)

        # n_u, n_m = train_r.shape
        # train_mask = (train_r > 0).astype(np.float32)
        # # Initialize models
        # kernel_net = KernelNet(n_u, self.n_hid, self.n_dim, self.n_layers, self.lambda_s, self.lambda_2).double().to(self.device)
        # complete_model = CompleteNet(kernel_net, n_u, n_m, self.n_hid, self.n_dim, self.n_layers, self.lambda_s, self.lambda_2, self.gk_size, self.dot_scale).double().to(self.device)

        self.min_rating = 1.0
        self.max_rating = 4.0

        # Extract user-item-rating tuples
        train_users, train_items, train_ratings = train_set.uir_tuple
        test_users, test_items, test_ratings = ([], [], [])  # For now, if val_set is None

        # Get total numbers of users and items
        n_u = train_set.num_users
        n_m = train_set.num_items

        # Construct train_r as (n_m, n_u), same as in your notebook
        train_r = np.zeros((n_m, n_u), dtype='float32')
        train_r[train_items, train_users] = train_ratings

        # Now train_r is shaped (n_m, n_u) and aligned with your model expectations.
        train_mask = (train_r > 0).astype(np.float32)


        self._train_r = train_r
        self._train_mask = train_mask



        # Update variables accordingly
        # After this, n_m = train_r.shape[0], n_u = train_r.shape[1]
        n_m, n_u = train_r.shape

        # Now initialize models with n_u and n_m that match the shape of train_r
        kernel_net = KernelNet(n_u, self.n_hid, self.n_dim, self.n_layers, self.lambda_s, self.lambda_2).double().to(self.device)
        complete_model = CompleteNet(kernel_net, n_u, n_m, self.n_hid, self.n_dim, self.n_layers, self.lambda_s, self.lambda_2, self.gk_size, self.dot_scale).double().to(self.device)


        # Pre-Training (KernelNet only)
        optimizer = torch.optim.AdamW(complete_model.local_kernel_net.parameters(), lr=self.lr_p)
        best_rmse = np.inf
        last_rmse = np.inf
        counter = 0

        tic = time.time()
        
        #Pre-training process with tqdm for every group of 10 epochs
        for group in range(0, self.max_epoch_p, 10):  # Split epochs into groups of 10
            start_epoch = group
            end_epoch = min(group + 10, self.max_epoch_p)  # Handle the last group

            # Initialize the progress bar for the group
            with tqdm(total=end_epoch - start_epoch, desc=f"Epochs {start_epoch + 1}-{end_epoch} (Pre-Training)", leave=True) as pbar:
                for epoch in range(start_epoch, end_epoch):

                    # Define the closure function
                    def closure():
                        optimizer.zero_grad()
                        x = torch.tensor(train_r, dtype=torch.double, device=self.device)
                        m = torch.tensor(train_mask, dtype=torch.double, device=self.device)
                        complete_model.local_kernel_net.train()
                        pred, reg = complete_model.local_kernel_net(x)
                        loss = Loss().to(self.device)(pred, reg, m, x)
                        loss.backward()
                        return loss

                    optimizer.step(closure)

                    complete_model.local_kernel_net.eval()
                    with torch.no_grad():
                        x = torch.tensor(train_r, dtype=torch.double, device=self.device)
                        pred, _ = kernel_net(x)

                    pre = pred.float().cpu().numpy()

                    # Compute training RMSE
                    train_rmse = np.sqrt(((train_mask * (np.clip(pre, 1., 5.) - train_r))**2).sum() / train_mask.sum())

                    # Update the current progress bar
                    pbar.set_postfix({"Train RMSE": f"{train_rmse:.4f}"})
                    pbar.update(1)

                    # Check for early stopping
                    if last_rmse - train_rmse < self.tol_p:
                        counter += 1
                    else:
                        counter = 0
                    last_rmse = train_rmse

                    if counter >= self.patience_p:
                        tqdm.write(f"Early stopping pre-training at epoch: {epoch + 1}")
                        break

                    # Log at the current epoch
                    if self.verbose:
                        logging.info(f"Pre-Training Epoch {epoch + 1}/{self.max_epoch_p}, Train RMSE: {train_rmse:.4f}")

        # After pre-training
        self.train_r_local = np.clip(pre, 1., 5.)

        # Fine-Tuning
        optimizer = torch.optim.AdamW(complete_model.parameters(), lr=self.lr_f)
        last_rmse = np.inf
        counter = 0
        
        for group in range(0, self.max_epoch_f, 10):  # Split epochs into groups of 10
            start_epoch = group
            end_epoch = min(group + 10, self.max_epoch_f)  # Handle the last group
            

            # Initialize the progress bar for the group
            with tqdm(total=end_epoch - start_epoch, desc=f"Epochs {start_epoch + 1}-{end_epoch}  (Fine-Tuning)", leave=True) as pbar:
                for epoch in range(start_epoch, end_epoch):
                    # Define the closure function
                    def closure():
                        optimizer.zero_grad()
                        x = torch.tensor(train_r, dtype=torch.double, device=self.device)
                        x_local = torch.tensor(self.train_r_local, dtype=torch.double, device=self.device)
                        m = torch.tensor(train_mask, dtype=torch.double, device=self.device)
                        complete_model.train()
                        pred, reg = complete_model(x, x_local)
                        loss = Loss().to(self.device)(pred, reg, m, x)
                        loss.backward()
                        return loss

                    optimizer.step(closure)

                    complete_model.eval()
                    with torch.no_grad():
                        x = torch.tensor(train_r, dtype=torch.double, device=self.device)
                        x_local = torch.tensor(self.train_r_local, dtype=torch.double, device=self.device)
                        pred, _ = complete_model(x, x_local)

                    pre = pred.float().cpu().numpy()

                    # Compute training RMSE
                    train_rmse = np.sqrt(((train_mask * (np.clip(pre, 1., 5.) - train_r))**2).sum() / train_mask.sum())

                    # Update the current progress bar
                    pbar.set_postfix({"Train RMSE": f"{train_rmse:.4f}"})
                    pbar.update(1)

                    # Check for early stopping
                    if last_rmse - train_rmse < self.tol_f:
                        counter += 1
                    else:
                        counter = 0
                    last_rmse = train_rmse

                    if counter >= self.patience_f:
                        tqdm.write(f"Early stopping fine-tuning at epoch: {epoch + 1}")
                        break

                    # Log at the current epoch
                    if self.verbose:
                        logging.info(f"Fine-Tuning Epoch {epoch + 1}/{self.max_epoch_f}, Train RMSE: {train_rmse:.4f}")

        # Store the trained model
        self.model = complete_model
        return self


    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item or batch of items.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.
        item_idx: int, optional
            The index (or indices) of the item(s) for which to perform score prediction.
            If None, scores for all items will be returned.

        Returns
        -------
        res: A scalar or Numpy array
            A scalar for a specific item, or a Numpy array of scores for all items.
        """
        if self.model is None:
            raise RuntimeError("You must train the model before calling score()!")

        with torch.no_grad():
            # Perform model predictions (full prediction matrix)
            input_mat = torch.tensor(self.train_r_local, dtype=torch.double, device=self.device)
            x_global = torch.tensor(self._train_r, dtype=torch.double, device=self.device)
            pred, _ = self.model(x_global, input_mat)
            pred = pred.cpu().numpy()  # Convert to NumPy array

        if item_idx is None:
            # Return scores for all items for the specified user
            return pred[:, user_idx]  # NumPy array of scores

        elif isinstance(item_idx, list):
            # Return scores for a list of items
            return np.array([pred[i, user_idx] for i in item_idx])  # NumPy array

        else:
            # Return score for a single item (scalar)
            return pred[item_idx, user_idx]





    # def get_vector_measure(self):
    #     from cornac.utils import MEASURE_DOT
    #     return MEASURE_DOT


    # def get_user_vectors(self):
    #     # Assuming self.U stores the user embeddings
    #     return self.U.cpu().detach().numpy()


    # def get_item_vectors(self):
    #     # Assuming self.V stores the item embeddings
    #     return self.V.cpu().detach().numpy()


    # def rank(self, user_idx, item_indices=None, k=None):
    #     """
    #     Rank items for a given user based on predicted scores.

    #     Parameters
    #     ----------
    #     user_idx : int
    #         The index of the user for whom to rank items.

    #     item_indices : array-like, optional, default: None
    #         Indices of items to be ranked. If None, rank all items.

    #     k : int, optional, default: None
    #         Number of top items to return. If None, return all ranked items.

    #     Returns
    #     -------
    #     item_rank : np.ndarray
    #         Indices of items ranked in descending order of predicted scores.

    #     item_scores : np.ndarray
    #         Predicted scores for the ranked items.
    #     """
    #     with torch.no_grad():
    #         # Get user embeddings (row from self.U)
    #         user_embedding = self.U[user_idx].cpu().numpy()
            
    #         # Compute scores for all items or a subset
    #         if item_indices is None:
    #             item_embeddings = self.V.cpu().numpy()  # All item embeddings
    #         else:
    #             item_embeddings = self.V[item_indices].cpu().numpy()  # Subset of items

    #         # Compute scores (dot product or similarity)
    #         scores = np.dot(item_embeddings, user_embedding)

    #         # Get the ranked indices
    #         ranked_indices = np.argsort(-scores)  # Descending order
    #         if k is not None:
    #             ranked_indices = ranked_indices[:k]

    #         # Get the corresponding scores for ranked items
    #         ranked_scores = scores[ranked_indices]

    #         # Map back to original item indices if item_indices is provided
    #         if item_indices is not None:
    #             ranked_indices = np.array(item_indices)[ranked_indices]

    #         return ranked_indices, ranked_scores
