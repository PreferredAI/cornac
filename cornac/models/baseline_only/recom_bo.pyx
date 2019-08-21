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

# cython: language_level=3

cimport cython
from cython.parallel import prange
from cython cimport floating, integral
from libcpp cimport bool
from libc.math cimport abs

import numpy as np
cimport numpy as np

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import fast_dot


class BaselineOnly(Recommender):
    """Baseline Only model uses user and item biases to estimate ratings.

    Parameters
    ----------
    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.01
        The learning rate.

    lambda_reg: float, optional, default: 0.001
        The lambda value used for regularization.

    early_stop: boolean, optional, default: False
        When True, delta loss will be checked after each iteration to stop learning earlier.

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'Bu': user_biases, 'Bi': item_biases}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Koren, Y. Factor in the neighbors: Scalable and accurate collaborative filtering. \
    In TKDD, 2010.
    """

    def __init__(self, name='BaselineOnly', max_iter=20, learning_rate=0.01, lambda_reg=0.02,
                 early_stop=False, num_threads=0, trainable=True, verbose=False, init_params=None, seed=None):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.early_stop = early_stop
        self.init_params = {} if init_params is None else init_params
        self.seed = seed

        import multiprocessing
        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

    def fit(self, train_set):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: object of type TrainSet, required
            An object contains the user-item preference in csr scipy sparse format,\
            as well as some useful attributes such as mappings to the original user/item ids.\
            Please refer to the class TrainSet in the "data" module for details.
        """
        Recommender.fit(self, train_set)

        n_users, n_items = train_set.num_users, train_set.num_items
        self.global_mean = train_set.global_mean

        from ...utils.init_utils import zeros
        self.u_biases = self.init_params.get('Bu', zeros(n_users))
        self.i_biases = self.init_params.get('Bi', zeros(n_items))

        if self.trainable:
            (rid, cid, val) = train_set.uir_tuple
            self._fit_sgd(rid, cid, val.astype(np.float32), self.u_biases, self.i_biases)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_sgd(self, integral[:] rid, integral[:] cid, floating[:] val, floating[:] Bu, floating[:] Bi):
        """Fit the model parameters (Bu, Bi) with SGD
        """
        cdef:
            long num_users = self.train_set.num_users
            long num_items = self.train_set.num_items
            long num_ratings = val.shape[0]
            int max_iter = self.max_iter
            int num_threads = self.num_threads

            floating reg = self.lambda_reg
            floating mu = self.global_mean

            bool early_stop = self.early_stop
            bool verbose = self.verbose

            floating lr = self.learning_rate
            floating loss = 0
            floating last_loss = 0
            floating r, r_pred, error, delta_loss
            integral u, i, j

        from tqdm import trange
        progress = trange(max_iter, disable=not self.verbose)
        for epoch in progress:
            last_loss = loss
            loss = 0

            for j in prange(num_ratings, nogil=True, schedule='static', num_threads=num_threads):
                u, i, r = rid[j], cid[j], val[j]

                r_pred = mu + Bu[u] + Bi[i]
                error = r - r_pred
                loss += error * error

                Bu[u] += lr * (error - reg * Bu[u])
                Bi[i] += lr * (error - reg * Bi[i])

            loss = 0.5 * loss
            progress.update(1)
            progress.set_postfix({"loss": "%.2f" % loss})

            delta_loss = loss - last_loss
            if early_stop and abs(delta_loss) < 1e-5:
                if verbose:
                    print('Early stopping, delta_loss = %.4f' % delta_loss)
                break
        progress.close()

        if verbose:
            print('Optimization finished!')


    def score(self, user_id, item_id=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score prediction.

        item_id: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        unk_user = self.train_set.is_unk_user(user_id)

        if item_id is None:
            known_item_scores = np.add(self.i_biases, self.global_mean)
            if not unk_user:
                known_item_scores = np.add(known_item_scores, self.u_biases[user_id])
            return known_item_scores
        else:
            unk_item = self.train_set.is_unk_item(item_id)
            item_score = self.global_mean
            if not unk_user:
                item_score += self.u_biases[user_id]
            if not unk_item:
                item_score += self.i_biases[item_id]
            return item_score
