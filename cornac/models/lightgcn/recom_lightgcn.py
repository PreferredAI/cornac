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
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException

from tqdm.auto import tqdm, trange


class LightGCN(Recommender, ANNMixin):
    """
    LightGCN

    Parameters
    ----------
    name: string, default: 'LightGCN'
        The name of the recommender model.

    emb_size: int, default: 64
        Size of the node embeddings.

    num_epochs: int, default: 1000
        Maximum number of iterations or the number of epochs.

    learning_rate: float, default: 0.001
        The learning rate that determines the step size at each iteration

    batch_size: int, default: 1024
        Mini-batch size used for train set

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
        emb_size=64,
        num_epochs=1000,
        learning_rate=0.001,
        batch_size=1024,
        num_layers=3,
        early_stopping=None,
        lambda_reg=1e-4,
        trainable=True,
        verbose=False,
        seed=2020,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.emb_size = emb_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_layers = num_layers
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        graph = construct_graph(train_set, self.total_users, self.total_items).to(
            device
        )
        model = Model(
            graph,
            self.emb_size,
            self.num_layers,
            self.lambda_reg,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

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
                    batch_size=self.batch_size,
                    shuffle=True,
                ),
                desc="Epoch",
                total=train_set.num_batches(self.batch_size),
                leave=False,
                position=1,
                disable=not self.verbose,
            ):
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(
                    graph, batch_u, batch_i, batch_j
                )

                batch_loss, batch_bpr_loss, batch_reg_loss = model.loss_fn(
                    u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
                )
                accum_loss += batch_loss.cpu().item() * len(batch_u)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            accum_loss /= len(train_set.uir_tuple[0])  # normalize over all observations
            pbar.set_postfix(loss=accum_loss)

            # store user and item embedding matrices for prediction
            model.eval()
            u_embs, i_embs, _ = model(graph)
            # we will use numpy for faster prediction in the score function, no need torch
            self.U = u_embs.cpu().detach().numpy()
            self.V = i_embs.cpu().detach().numpy()

            if self.early_stopping is not None and self.early_stop(
                train_set, val_set, **self.early_stopping
            ):
                break

    def monitor_value(self, train_set, val_set):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        res : float
            Monitored value on validation set.
            Return `None` if `val_set` is `None`.
        """
        if val_set is None:
            return None

        from ...metrics import Recall
        from ...eval_methods import ranking_eval

        recall_20 = ranking_eval(
            model=self,
            metrics=[Recall(k=20)],
            train_set=train_set,
            test_set=val_set,
        )[0][0]

        return recall_20  # Section 4.1.2 in the paper, same strategy as NGCF.

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
            if not self.knows_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            known_item_scores = self.V.dot(self.U[user_idx, :])
            return known_item_scores
        else:
            if not (self.knows_user(user_idx) and self.knows_item(item_idx)):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            return self.V[item_idx, :].dot(self.U[user_idx, :])

    def get_vector_measure(self):
        """Getting a valid choice of vector measurement in ANNMixin._measures.

        Returns
        -------
        measure: MEASURE_DOT
            Dot product aka. inner product
        """
        return MEASURE_DOT

    def get_user_vectors(self):
        """Getting a matrix of user vectors serving as query for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of user vectors for all users available in the model.
        """
        return self.U

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model.
        """
        return self.V
