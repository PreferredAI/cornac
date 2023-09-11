import logging
import time
import torch
import numpy as np
from .data import construct_graph
from .nn_modules import NeuralNetwork
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Model:
    def __init__(
        self,
        num_users,
        num_items,
        hidden_dim,
        num_layers,
        learning_rate,
        batch_size,
        verbose,
        seed,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = NeuralNetwork(
            num_users,
            num_items,
            hidden_dim,
            num_layers,
            self.device
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0)

        self.verbose = verbose
        if verbose:
            logging.basicConfig(level=logging.INFO)

        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.best_valid_loss = np.inf
        self.writer = SummaryWriter()

    def _calculate_loss(self, batch_u, batch_i, batch_j, user_embeddings, item_embeddings):        
        user_embed = user_embeddings[batch_u]
        positive_item_embed = item_embeddings[batch_i]
        negative_item_embed = item_embeddings[batch_j]

        pred_i = torch.sum(
            torch.multiply(user_embed, positive_item_embed), dim=1
        )
        pred_j = torch.sum(
            torch.multiply(user_embed, negative_item_embed), dim=1
        )

        # bpr_loss = - (pred_j.view(-1) - pred_i.view(-1)).sigmoid().log().sum()
        # bpr_loss = torch.mean(torch.nn.functional.softplus(pred_j - pred_i))

        # reg_loss = (
        #     torch.norm(user_embed) ** 2 +
        #     torch.norm(positive_item_embed) ** 2 +
        #     torch.norm(negative_item_embed) ** 2
        # )

        reg_loss = (1/2) * (
            user_embed.norm(2).pow(2) +
            positive_item_embed.norm(2).pow(2) +
            negative_item_embed.norm(2).pow(2)
        ) / float(len(batch_u))

        loss = torch.mean(torch.nn.functional.softplus(pred_j - pred_i))
        # loss = 0.5 * (bpr_loss + self.learning_rate * reg_loss) / self.batch_size

        return loss, reg_loss

    def _train_model(self, train_set, val_set, model, graph, weight_decay):
        epoch_loss_test = 0
        epoch_loss_val = None

        # Train set
        for batch_u, batch_i, batch_j in tqdm(
            train_set.uij_iter(
                batch_size=self.batch_size,
                shuffle=True,
            ),
            desc="Batch",
            total=train_set.num_batches(self.batch_size),
            leave=False,
            position=1
        ):
            batch_u = torch.from_numpy(batch_u).long().to(self.device)
            batch_i = torch.from_numpy(batch_i).long().to(self.device)
            batch_j = torch.from_numpy(batch_j).long().to(self.device)

            user_embeddings, item_embeddings = model(graph)
            loss, reg_loss = self._calculate_loss(batch_u, batch_i, batch_j, user_embeddings, item_embeddings)
            reg_loss = reg_loss * weight_decay
            loss = loss + reg_loss
            epoch_loss_test += loss.cpu().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # user_embed, pos_item_embed, neg_item_embed = _get_embedding(self, users, )

            # for u, i, j in zip(batch_u, batch_i, batch_j):
            #     user_embeddings, item_embeddings = model(graph)
            #     bpr_loss, loss = self._calculate_loss(u, i, j, user_embeddings, item_embeddings)
            #     epoch_loss_test += bpr_loss.item()
                
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.optimizer.step()
        
        # # Val set
        # if val_set is not None:
        #     epoch_loss_val = 0

        #     for batch_u, batch_i, batch_j in tqdm(
        #         val_set.uij_iter(
        #             batch_size=self.batch_size,
        #             shuffle=True,
        #         ),
        #         desc="Val Set",
        #         total=val_set.num_batches(self.batch_size),
        #         leave=False,
        #         position=1
        #     ):
        #         batch_u = torch.from_numpy(batch_u).long().to(self.device)
        #         batch_i = torch.from_numpy(batch_i).long().to(self.device)
        #         batch_j = torch.from_numpy(batch_j).long().to(self.device)

        #         user_embeddings, item_embeddings = model(graph)
        #         bpr_loss, _ = self._calculate_loss(batch_u, batch_i, batch_j, user_embeddings, item_embeddings)
        #         epoch_loss_val += bpr_loss.item()

        #     if epoch_loss_val < self.best_valid_loss:
        #         self.best_valid_loss = epoch_loss_val
        #         self.no_better_valid_count = 0
        #         self.best_model_state_dict = model.state_dict()

        epoch_loss_test = epoch_loss_test/self.batch_size

        return epoch_loss_test, epoch_loss_val

    def train(
        self,
        train_set,
        val_set,
        max_iter,
        weight_decay
    ):
        self.graph = construct_graph(train_set, self.device)
        # progress_bar = trange(1, max_iter + 1, disable=self.verbose)
        for iter in tqdm(
            range(1, max_iter),
            desc="Training",
            unit="iter",
            position=0,
            leave=False,
            # disable=self.verbose,
        ):
            loss_test, loss_val = self._train_model(train_set, val_set, self.model, self.graph, weight_decay)

            log_str = f"Epoch: {iter}\t loss: {loss_test:.4f}"
            if loss_val is not None:
                log_str += f"\t val_loss: {loss_val:.4f}"
            logging.info(log_str)

            self.writer.add_scalar("Loss/train", loss_test, iter)

    def predict(
        self,
        test_set
    ):
        graph = construct_graph(test_set, self.device)
        user_embeddings, item_embeddings = self.model(graph)
        return user_embeddings, item_embeddings

    def score(
        self,
        user_embeddings,
        item_embeddings,
        user_idx,
        item_idx
    ):
        if item_idx is None:
            return torch.matmul(item_embeddings, torch.t(user_embeddings[user_idx])).cpu().detach().numpy()
        else:
            return torch.sum(torch.multiply(user_embeddings[user_idx], item_embeddings[item_idx])).cpu().detach().numpy()
        
    def monitor_value(self):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.

        Returns
        -------
        res : float
            Monitored value on validation set.
            Return `None` if `val_set` is `None`.
        """
        if self.val_set is None:
            return None
        
        epoch_loss_val = 0
        for batch_u, batch_i, batch_j in tqdm(
            self.val_set.uij_iter(
                batch_size=self.batch_size,
                shuffle=True,
            ),
            desc="Val Set",
            total=self.val_set.num_batches(self.batch_size),
            leave=False,
            position=1
        ):
            batch_u = torch.from_numpy(batch_u).long().to(self.device)
            batch_i = torch.from_numpy(batch_i).long().to(self.device)
            batch_j = torch.from_numpy(batch_j).long().to(self.device)

            user_embeddings, item_embeddings = self.model(self.graph)
            bpr_loss, _ = self._calculate_loss(batch_u, batch_i, batch_j, user_embeddings, item_embeddings)
            epoch_loss_val += bpr_loss.item()

        logging.info(f"loss_val: {epoch_loss_val}")

        return epoch_loss_val

        

