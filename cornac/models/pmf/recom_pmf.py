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


class PMF(Recommender):
    """Probabilistic Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 5
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD_RMSProp.
        
    gamma: float, optional, default: 0.9
        The weight for previous/current gradient in RMSProp.

    lambda_reg: float, optional, default: 0.001
        The regularization coefficient.

    name: string, optional, default: 'PMF'
        The name of the recommender model.
        
    variant: {"linear","non_linear"}, optional, default: 'non_linear'
        Pmf variant. If 'non_linear', the Gaussian mean is the output of a Sigmoid function.\
        If 'linear' the Gaussian mean is the output of the identity function.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dict, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}.
        
        U: ndarray, shape (n_users, k) 
            User latent factors.
        
        V: ndarray, shape (n_items, k)
            Item latent factors.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Mnih, Andriy, and Ruslan R. Salakhutdinov. Probabilistic matrix factorization. \
    In NIPS, pp. 1257-1264. 2008.
    """

    def __init__(
        self,
        k=5,
        max_iter=100,
        learning_rate=0.001,
        gamma=0.9,
        lambda_reg=0.001,
        name="PMF",
        variant="non_linear",
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.variant = variant
        self.seed = seed

        self.ll = np.full(max_iter, 0)
        self.eps = 0.000000001
        
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
        Recommender.fit(self, train_set)

        from cornac.models.pmf import pmf

        if self.trainable:
            # converting data to the triplet format (needed for cython function pmf)
            (uid, iid, rat) = train_set.uir_tuple
            rat = np.array(rat, dtype="float32")
            if self.variant == "non_linear":  # need to map the ratings to [0,1]
                if [self.train_set.min_rating, self.train_set.max_rating] != [0, 1]:
                    rat = scale(
                        rat,
                        0.0,
                        1.0,
                        self.train_set.min_rating,
                        self.train_set.max_rating,
                    )
            uid = np.array(uid, dtype="int32")
            iid = np.array(iid, dtype="int32")

            if self.verbose:
                print("Learning...")
                
            # use pre-trained params if exists, otherwise from constructor
            init_params = {"U": self.U, "V": self.V}

            if self.variant == "linear":
                res = pmf.pmf_linear(
                    uid,
                    iid,
                    rat,
                    k=self.k,
                    n_users=train_set.num_users,
                    n_items=train_set.num_items,
                    n_ratings=len(rat),
                    n_epochs=self.max_iter,
                    lambda_reg=self.lambda_reg,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    init_params=init_params,
                    verbose=self.verbose,
                    seed=self.seed,
                )
            elif self.variant == "non_linear":
                res = pmf.pmf_non_linear(
                    uid,
                    iid,
                    rat,
                    k=self.k,
                    n_users=train_set.num_users,
                    n_items=train_set.num_items,
                    n_ratings=len(rat),
                    n_epochs=self.max_iter,
                    lambda_reg=self.lambda_reg,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    init_params=init_params,
                    verbose=self.verbose,
                    seed=self.seed,
                )
            else:
                raise ValueError('variant must be one of {"linear","non_linear"}')

            self.U = np.asarray(res["U"])
            self.V = np.asarray(res["V"])

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

            user_pred = self.V[item_idx, :].dot(self.U[user_idx, :])

            if self.variant == "non_linear":
                user_pred = sigmoid(user_pred)
                user_pred = scale(
                    user_pred,
                    self.train_set.min_rating,
                    self.train_set.max_rating,
                    0.0,
                    1.0,
                )

            return user_pred
