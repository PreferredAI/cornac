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
from ...utils.common import sigmoid
from ...utils.common import scale
from ...exception import ScoreException


class VMF(Recommender):
    """Visual Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the user and item factors.
        
    d: int, optional, default: 10
       The dimension of the user visual factors.

    n_epochs: int, optional, default: 100
        The number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD_RMSProp.

    gamma: float, optional, default: 0.9
        The weight for previous/current gradient in RMSProp.

    lambda_u: float, optional, default: 0.001
        The regularization parameter for user factors.

    lambda_v: float, optional, default: 0.001
        The regularization parameter for item factors.

    lambda_p: float, optional, default: 1.0
        The regularization parameter for user visual factors.

    lambda_e: float, optional, default: 10.
        The regularization parameter for the kernel embedding matrix
        
    lambda_u: float, optional, default: 0.001
        The regularization parameter for user factors.

    name: string, optional, default: 'VMF'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained (The parameters of the model U, V, P, E are not None).
        
    visual_features: See "cornac/examples/vmf_example.py" for an example of how to use \
        cornac's visual modality to load and provide the "item visual features" for VMF.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V, 'P': P, 'E': E}.

        U: numpy array of shape (n_users,k), user latent factors.
        V: numpy array of shape (n_items,k), item latent factors.
        P: numpy array of shape (n_users,d), user visual latent factors.
        E: numpy array of shape (d,c), embedding kernel matrix.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Park, Chanyoung, Donghyun Kim, Jinoh Oh, and Hwanjo Yu. "Do Also-Viewed Products Help User Rating Prediction?."\
     In Proceedings of WWW, pp. 1113-1122. 2017.

    """

    def __init__(
        self,
        name="VMF",
        k=10,
        d=10,
        n_epochs=100,
        batch_size=100,
        learning_rate=0.001,
        gamma=0.9,
        lambda_u=0.001,
        lambda_v=0.001,
        lambda_p=1.0,
        lambda_e=10.0,
        trainable=True,
        verbose=False,
        use_gpu=False,
        init_params=None,
        seed=None,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.d = d
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_p = lambda_p
        self.lambda_e = lambda_e
        self.use_gpu = use_gpu
        self.loss = np.full(n_epochs, 0)
        self.eps = 0.000000001
        self.seed = seed

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)  # user factors
        self.V = self.init_params.get("V", None)  # item factors
        self.P = self.init_params.get("P", None)  # user visual factors
        self.E = self.init_params.get("E", None)  # Kernel embedding matrix

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
            # Item visual cnn-features
            item_features = train_set.item_image.features[: self.train_set.num_items]

            if self.verbose:
                print("Learning...")

            from .vmf import vmf

            res = vmf(
                self.train_set,
                item_features,
                k=self.k,
                d=self.d,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                lambda_u=self.lambda_u,
                lambda_v=self.lambda_v,
                lambda_p=self.lambda_p,
                lambda_e=self.lambda_e,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                init_params={"U": self.U, "V": self.V, "P": self.P, "E": self.E},
                use_gpu=self.use_gpu,
                verbose=self.verbose,
                seed=self.seed,
            )

            self.U = res["U"]
            self.V = res["V"]
            self.P = res["P"]
            self.E = res["E"]
            self.Q = res["Q"]

            if self.verbose:
                print("Learning completed")

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

        return self

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

            known_item_scores = self.V.dot(self.U[user_idx, :]) + self.Q.dot(
                self.P[user_idx, :]
            )
            # known_item_scores = np.asarray(np.zeros(self.V.shape[0]),dtype='float32')
            # fast_dot(self.U[user_id], self.V, known_item_scores)
            # fast_dot(self.P[user_id], self.Q, known_item_scores)
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            user_pred = self.V[item_idx, :].dot(self.U[user_idx, :]) + self.Q[
                item_idx, :
            ].dot(self.P[user_idx, :])
            user_pred = sigmoid(user_pred)

            user_pred = scale(
                user_pred,
                self.train_set.min_rating,
                self.train_set.max_rating,
                0.0,
                1.0,
            )

            return user_pred
