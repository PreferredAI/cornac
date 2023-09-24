# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from ..recommender import Recommender
from ...exception import ScoreException

from tqdm.auto import tqdm, trange


class LightGCN(Recommender):
    """
    LightGCN

    Parameters
    ----------
    name: string, default: 'LightGCN'
        The name of the recommender model.

    num_epochs: int, default: 1000
        Maximum number of iterations or the number of epochs

    learning_rate: float, default: 0.001
        The learning rate that determines the step size at each iteration

    train_batch_size: int, default: 1024
        Mini-batch size used for train set

    test_batch_size: int, default: 100
        Mini-batch size used for test set

    hidden_dim: int, default: 64
        The embedding size of the model

    num_layers: int, default: 3
        Number of LightGCN Layers

    early_stopping: {min_delta: float, patience: int}, optional, default: None
        If `None`, no early stopping. Meaning of the arguments:

        - `min_delta`:  the minimum increase in monitored value on validation
                        set to be considered as improvement,
                        i.e. an increment of less than min_delta will count as
                        no improvement.

        - `patience`:   number of epochs with no improvement after which
                        training should be stopped.

    lambda_reg: float, default: 1e-4
        Weight decay for the L2 normalization

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model
        is already pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: 2020
        Random seed for parameters initialization.

    References
    ----------
    *   He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020).
        LightGCN: Simplifying and Powering Graph Convolution Network for
        Recommendation.
    """

    def __init__(
        self,
        name="LightGCN",
        num_epochs=1000,
        learning_rate=0.001,
        train_batch_size=1024,
        test_batch_size=100,
        hidden_dim=64,
        num_layers=3,
        early_stopping=None,
        lambda_reg=1e-4,
        trainable=True,
        verbose=False,
        seed=2020,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.early_stopping = early_stopping
        self.lambda_reg = lambda_reg
        self.seed = seed

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if not self.trainable:
            return self

        # model setup
        import torch
        from .lightgcn import Model
        from .lightgcn import construct_graph

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        model = Model(
            train_set.total_users,
            train_set.total_items,
            self.hidden_dim,
            self.num_layers,
        ).to(self.device)

        graph = construct_graph(train_set).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.lambda_reg
        )
        loss_fn = torch.nn.BCELoss(reduction="sum")

        # model training
        pbar = trange(
            self.num_epochs,
            desc="Training",
            unit="iter",
            position=0,
            leave=False,
            disable=not self.verbose,
        )
        for _ in pbar:
            model.train()
            accum_loss = 0.0
            for batch_u, batch_i, batch_j in tqdm(
                train_set.uij_iter(
                    batch_size=self.train_batch_size,
                    shuffle=True,
                ),
                desc="Epoch",
                total=train_set.num_batches(self.train_batch_size),
                leave=False,
                position=1,
                disable=not self.verbose,
            ):
                user_embeddings, item_embeddings = model(graph)

                batch_u = torch.from_numpy(batch_u).long().to(self.device)
                batch_i = torch.from_numpy(batch_i).long().to(self.device)
                batch_j = torch.from_numpy(batch_j).long().to(self.device)

                user_embed = user_embeddings[batch_u]
                positive_item_embed = item_embeddings[batch_i]
                negative_item_embed = item_embeddings[batch_j]

                ui_scores = (user_embed * positive_item_embed).sum(dim=1)
                uj_scores = (user_embed * negative_item_embed).sum(dim=1)

                loss = loss_fn(
                    torch.sigmoid(ui_scores - uj_scores), torch.ones_like(ui_scores)
                )
                accum_loss += loss.cpu().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accum_loss /= len(train_set.uir_tuple[0])  # normalize over all observations
            pbar.set_postfix(loss=accum_loss)

            # store user and item embedding matrices for prediction
            model.eval()
            self.U, self.V = model(graph)

            if self.early_stopping is not None and self.early_stop(
                **self.early_stopping
            ):
                break

        # we will use numpy for faster prediction in the score function, no need torch
        self.U = self.U.cpu().detach().numpy()
        self.V = self.V.cpu().detach().numpy()

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

        import torch

        loss_fn = torch.nn.BCELoss(reduction="sum")
        accum_loss = 0.0
        pbar = tqdm(
            self.val_set.uij_iter(batch_size=self.test_batch_size),
            desc="Validation",
            total=self.val_set.num_batches(self.test_batch_size),
            leave=False,
            position=1,
            disable=not self.verbose,
        )
        for batch_u, batch_i, batch_j in pbar:
            batch_u = torch.from_numpy(batch_u).long().to(self.device)
            batch_i = torch.from_numpy(batch_i).long().to(self.device)
            batch_j = torch.from_numpy(batch_j).long().to(self.device)

            user_embed = self.U[batch_u]
            positive_item_embed = self.V[batch_i]
            negative_item_embed = self.V[batch_j]

            ui_scores = (user_embed * positive_item_embed).sum(dim=1)
            uj_scores = (user_embed * negative_item_embed).sum(dim=1)

            loss = loss_fn(
                torch.sigmoid(ui_scores - uj_scores), torch.ones_like(ui_scores)
            )
            accum_loss += loss.cpu().item()
            pbar.set_postfix(val_loss=accum_loss)

        accum_loss /= len(self.val_set.uir_tuple[0])
        return -accum_loss  # higher is better -> smaller loss is better

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            known_item_scores = self.V.dot(self.U[user_idx, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            return self.V[item_idx, :].dot(self.U[user_idx, :])
