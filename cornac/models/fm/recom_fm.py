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


import sys
import multiprocessing

import numpy as np

from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...utils import get_rng
from ...utils.init_utils import zeros, normal


class FM(Recommender, ANNMixin):
    """Factorization Machines.

    Parameters
    ----------
    k0: int, optional, default: 1
        Using bias 'w0'.
    
    k1: int, optional, default: 1
        Using first-order weights 'w'.

    k2: int, optional, default: 8
        Dimension of second-order weights 'v'.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD, SGDA.

    learning_rate: float, optional, default: 0.01
        The learning rate for SGD.

    reg0: float, optional, default: 0.0
        Bias regularization. 

    reg1: float, optional, default: 0.0
        First-order weights regularization. 

    reg2: float, optional, default: 0.0
        Second-order weights regularization. 

    reg_all: float, optional, default: 0.0
        Regularization for all parameters.
        If 'reg_all' != 0.0, this will be used for all regularization.

    init_stdev: float, optional, default: 0.1
        Standard deviation for initialization of second-order weights 'v'.

    method, str, optional, default: 'mcmc'
        Learning method (SGD, SGDA, ALS, MCMC)

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'w0': bias, 'w': first-order, 'v': second-order}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Rendle, S. (2010, December). Factorization machines. \
    In 2010 IEEE International Conference on Data Mining (pp. 995-1000). IEEE.
    
    * Rendle, S. (2012). Factorization machines with libfm. \
    ACM Transactions on Intelligent Systems and Technology (TIST), 3(3), 1-22.
    """

    def __init__(
        self,
        name="FM",
        k0=1,
        k1=1,
        k2=8,
        max_iter=100,
        learning_rate=0.01,
        reg0=0.0,
        reg1=0.0,
        reg2=0.0,
        reg_all=0.0,
        init_stdev=0.1,
        method="mcmc",
        num_threads=0,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        if not sys.platform.startswith("linux"):
            exit(
                "FM model is only supported on Linux.\n"
                + "Windows executable can be found at http://www.libfm.org."
            )

        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        self.reg0 = reg_all if reg_all != 0.0 else reg0
        self.reg1 = reg_all if reg_all != 0.0 else reg1
        self.reg2 = reg_all if reg_all != 0.0 else reg2

        self.init_stdev = init_stdev
        self.method = method.lower()
        self.seed = seed
        self.rng = get_rng(seed)

        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.w0 = self.init_params.get("w0", None)
        self.w = self.init_params.get("w", None)
        self.v = self.init_params.get("v", None)

    def _init(self):
        num_features = self.total_users + self.total_items

        if self.w0 is None:
            self.w0 = 0.0
        if self.w is None:
            self.w = zeros(num_features, dtype=np.double)
        if self.v is None:
            d = self.k2 if self.k2 else 1  # dummy if self.k2 == 0
            self.v = normal(
                (d, num_features),
                std=self.init_stdev,
                random_state=self.seed,
                dtype=np.double,
            )

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

        self._init()

        if not self.trainable:
            return self

        from cornac.models.fm import backend_libfm

        self.w0 = backend_libfm.learn(
            train_set,
            val_set,
            self.w0,
            self.w,
            self.v,
            self.total_users,
            self.total_items,
            self.k0,
            self.k1,
            self.k2,
            self.max_iter,
            self.learning_rate,
            self.reg0,
            self.reg1,
            self.reg2,
            self.min_rating,
            self.max_rating,
            self.init_stdev,
            self.method,
            self.verbose,
        )

        if self.verbose:
            print("Optimization finished!")

        return self

    def _fm_predict(self, user_idx, item_idx):
        uid = user_idx
        iid = item_idx + self.total_users
        score = 0.0
        if self.k0:
            score += self.w0
        if self.k1:
            score += self.w[uid] + self.w[iid]
        if self.k2:
            score += self.v[:, uid].dot(self.v[:, iid])
        return score

    def _fm_predict_all(self, user_idx):
        uid = user_idx
        iid_start = self.total_users
        scores = np.zeros(self.total_items)
        if self.k0:
            scores += self.w0
        if self.k1:
            scores += self.w[uid] + self.w[iid_start:]
        if self.k2:
            sum_ = self.v[:, uid, np.newaxis] + self.v[:, iid_start:]
            sum_sqr_ = self.v[:, uid, np.newaxis] ** 2 + self.v[:, iid_start:] ** 2
            scores += 0.5 * (sum_**2 - sum_sqr_).sum(0)
        return scores

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            return self._fm_predict_all(user_idx)
        else:
            return self._fm_predict(user_idx, item_idx)

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
        user_vectors = self.v[:, : self.total_users]
        if self.k1:  # has bias term
            user_vectors = np.concatenate(
                (user_vectors, np.ones([user_vectors.shape[0], 1])), axis=1
            )
        return user_vectors

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model.
        """
        item_vectors = self.v[:, self.total_users :]
        if self.k1:  # has bias term
            item_vectors = np.concatenate(
                (item_vectors, self.w[self.total_users :].reshape((-1, 1))), axis=1
            )
        return item_vectors
