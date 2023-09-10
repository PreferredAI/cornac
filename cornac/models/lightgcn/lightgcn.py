import logging
import torch
import numpy as np
from .data import construct_graph
from .nn_modules import NeuralNetwork
from tqdm import tqdm


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

    def _train_model(self, train_set, model, graph):
        epoch_loss = 0
        for batch_u, batch_i, batch_j in train_set.uij_iter(
            batch_size=self.batch_size,
            shuffle=True,
        ):
            for u, i, j in tqdm(zip(batch_u, batch_i, batch_j), desc="Batch Item", total=self.batch_size, leave=False, position=1):
                u = torch.from_numpy(np.asarray(u)).long().to(self.device)
                i = torch.from_numpy(np.asarray(i)).long().to(self.device)
                j = torch.from_numpy(np.asarray(j)).long().to(self.device)

                user_embeddings, item_embeddings = model(graph)

                user_embed = user_embeddings[u]
                positive_item_embed = item_embeddings[i]
                negative_item_embed = item_embeddings[j]

                pred_i = torch.sum(
                    torch.multiply(user_embed, positive_item_embed)
                )
                pred_j = torch.sum(
                    torch.multiply(user_embed, negative_item_embed)
                )

                bpr_loss = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log().sum()
                reg_loss = (
                    torch.norm(user_embed) ** 2 +
                    torch.norm(positive_item_embed) ** 2 +
                    torch.norm(negative_item_embed) ** 2
                )

                loss = 0.5 * (bpr_loss + self.learning_rate * reg_loss) / self.batch_size

                epoch_loss += bpr_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            break  # mini batching, we only want the first batch

        return epoch_loss

    def train(
        self,
        train_set,
        val_set,
        max_iter
    ):
        graph = construct_graph(train_set, self.device)
        for iter in tqdm(
            range(1, max_iter),
            desc="Training",
            unit="iter",
            disable=self.verbose,
        ):
            loss = self._train_model(train_set, self.model, graph)
            logging.info(
                f"Epoch: {iter}\t"
                + f"loss: {loss:.4f}"
            )

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
