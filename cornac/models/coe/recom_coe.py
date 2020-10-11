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

import numpy as np

from ..recommender import Recommender
from ...exception import ScoreException


class COE(Recommender):
    """Collaborative Ordinal Embedding.

    Parameters
    ----------
    k: int, optional, default: 20
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.05
        The learning rate for SGD.

    lamda: float, optional, default: 0.001
        The regularization parameter.

    batch_size: int, optional, default: 100
        The batch size for SGD.

    name: string, optional, default: 'IBRP'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}.

        U: ndarray, shape (n_users, k)
            The user latent factors.

        V: ndarray, shape (n_items, k)
            The item latent factors.

    References
    ----------
    * Le, D. D., & Lauw, H. W. (2016, June). Euclidean co-embedding of ordinal data for multi-type visualization.\
     In Proceedings of the 2016 SIAM International Conference on Data Mining (pp. 396-404). Society for Industrial and Applied Mathematics.
    """

    def __init__(
        self,
        k=20,
        max_iter=100,
        learning_rate=0.05,
        lamda=0.001,
        batch_size=1000,
        name="coe",
        trainable=True,
        verbose=False,
        init_params=None,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.name = name
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.batch_size = batch_size

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)  # matrix of user factors
        self.V = self.init_params.get("V", None)  # matrix of item factors

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

        if self.trainable:
            from .coe import coe

            if self.verbose:
                print("Learning...")

            res = coe(
                self.train_set.matrix,
                k=self.k,
                n_epochs=self.max_iter,
                lamda=self.lamda,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                init_params={"U": self.U, "V": self.V},
            )
            self.U = np.asarray(res["U"])
            self.V = np.asarray(res["V"])

            if self.verbose:
                print("Learning completed")

        return self

    # get prefiction for a single user (predictions for one user at a time for efficiency purposes)
    # predictions are not stored for the same efficiency reasons"""

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

            known_item_scores = np.sum(
                np.abs(self.V - self.U[user_idx, :]) ** 2, axis=-1
            ) ** (1.0 / 2)
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = np.sum(
                np.abs(self.V[item_idx, :] - self.U[user_idx, :]) ** 2, axis=-1
            ) ** (1.0 / 2)
            return user_pred
