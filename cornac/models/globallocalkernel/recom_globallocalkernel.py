import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from ..recommender import Recommender

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

        nn.init.xavier_uniform_(self.W, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.u, gain=torch.nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.v, gain=torch.nn.init.calculate_gain("relu"))
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
        n_hid=10,       # size of hidden layers
        n_dim=2,         # inner AE embedding size
        n_layers=2,      # number of hidden layers
        lambda_s=0.006,  # regularization of sparsity of the final matrix
        lambda_2=20.,    # regularization of number of parameters
        gk_size=3,       # width=height of kernel for convolution
        dot_scale=1,     # dot product weight for global kernel
        max_epoch_p=2, # max number of epochs for pretraining
        max_epoch_f=2, # max number of epochs for finetuning
        tol_p=1e-2,      # min threshold for difference between consecutive values of train rmse for pretraining
        tol_f=1e-2,      # min threshold for difference for finetuning
        patience_p=1,    # early stopping patience for pretraining
        patience_f=1,   # early stopping patience for finetuning
        lr=0.001, 
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
        self.lr = lr
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
        self.max_rating = 5.0

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
        optimizer = torch.optim.AdamW(complete_model.local_kernel_net.parameters(), lr=self.lr)
        best_rmse = np.inf
        last_rmse = np.inf
        counter = 0

        tic = time.time()

        for epoch in range(self.max_epoch_p):
            # def closure():
            #     optimizer.zero_grad()
            #     x = torch.tensor(train_r, dtype=torch.double, device=self.device)
            #     m = torch.tensor(train_mask, dtype=torch.double, device=self.device)
            #     complete_model.local_kernel_net.train()
            #     pred, reg = complete_model.local_kernel_net(x)
            #     loss = Loss().to(self.device)(pred, reg, m, x)
            #     loss.backward()
            #     return loss


            def closure():
                optimizer.zero_grad()
                # Use train_r instead of train_r
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
                # print('complete model train_r :' , train_r)
                x = torch.tensor(train_r, dtype=torch.double, device=self.device)
                pred, _ = kernel_net(x)
            pre = pred.float().cpu().numpy()
            # Compute training RMSE
            train_rmse = np.sqrt(((train_mask * (np.clip(pre, 1., 5.) - train_r))**2).sum() / train_mask.sum())

            if last_rmse - train_rmse < self.tol_p:
                counter += 1
            else:
                counter = 0
            last_rmse = train_rmse

            if counter >= self.patience_p:
                if self.verbose:
                    print("Early stopping pre-training at epoch:", epoch+1)
                break

            if self.verbose and epoch % 10 == 0:
                print(f"Pre-Training Epoch {epoch+1}/{self.max_epoch_p}, Train RMSE: {train_rmse:.4f}")

        # After pre-training
        self.train_r_local = np.clip(pre, 1., 5.)

        # Fine-Tuning
        optimizer = torch.optim.AdamW(complete_model.parameters(), lr=self.lr)
        last_rmse = np.inf
        counter = 0

        for epoch in range(self.max_epoch_f):
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

            if last_rmse - train_rmse < self.tol_f:
                counter += 1
            else:
                counter = 0
            last_rmse = train_rmse

            if counter >= self.patience_f:
                if self.verbose:
                    print("Early stopping fine-tuning at epoch:", epoch+1)
                break

            if self.verbose and epoch % 10 == 0:
                print(f"Fine-Tuning Epoch {epoch+1}/{self.max_epoch_f}, Train RMSE: {train_rmse:.4f}")

        # Store the trained model
        self.model = complete_model
        return self

    def score(self, user_id, item_id=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score prediction.

        item_id: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
        """
        if self.model is None:
            raise RuntimeError("You must train the model before calling score()!")

        # Inference: provide predictions for given user_id, item_id
        # We'll assume we've stored training rating matrix in `fit` if needed.
        # For simplicity, assume `train_r_local` and `model` are available.

        # Note: For large datasets, keep user and item embeddings precomputed.
        with torch.no_grad():
            # We can re-use self.train_r_local as input
            # If user_id given, we create a vector with only that user
            # We'll do a full prediction for all users/items and slice.
            # In a production scenario, you'd probably have a more efficient inference method.

            # self.model expects full matrix input:
            # Construct a matrix with shape (num_users, num_items), with zeros if needed
            # For scoring, you can either store the training data or create a neutral input
            # Here we just re-use training data as input context.
            input_mat = torch.tensor(self.train_r_local, dtype=torch.double, device=self.device)
            # We must pass also the original train_r for global kernel step:
            # If we have it stored somewhere, we should keep it. Here we assume we have them:
            # Ideally, you might store self.train_r in self.fit for scoring:
            # For demonstration, let's assume we stored train_r in self._train_r
            x_global = torch.tensor(self._train_r, dtype=torch.double, device=self.device)
            # Compute predictions
            pred, _ = self.model(x_global, input_mat)
            pred = pred.float().cpu().numpy()

        if item_id is None:
            # return predictions for all items for this user
            return pred[:, user_id]
        else:
            # return prediction for this single item
            return pred[item_id , user_id]

    def rate(self, user_id, item_id):
        # Optionally override if needed, or rely on default Recommender.rate()
        return super().rate(user_id, item_id)
