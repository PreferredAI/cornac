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

import multiprocessing

cimport cython
from cython.parallel import prange
from cython cimport floating, integral
from libcpp cimport bool
from libc.math cimport abs

import numpy as np
cimport numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException
from ...utils import fast_dot
from ...utils import get_rng
from ...utils.init_utils import normal, zeros



class MF(Recommender, ANNMixin):
    """Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.01
        The learning rate.

    lambda_reg: float, optional, default: 0.001
        The lambda value used for regularization.

    use_bias: boolean, optional, default: True
        When True, user, item, and global biases are used.

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
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors,
        'Bu': user_biases, 'Bi': item_biases}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Koren, Y., Bell, R., & Volinsky, C. Matrix factorization techniques for recommender systems. \
    In Computer, (8), 30-37. 2009.
    """

    def __init__(
        self, 
        name='MF', 
        k=10, 
        max_iter=20, 
        learning_rate=0.01, 
        lambda_reg=0.02, 
        use_bias=True,
        early_stop=False, 
        num_threads=0, 
        trainable=True, 
        verbose=False, 
        init_params=None, 
        seed=None
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.use_bias = use_bias
        self.early_stop = early_stop
        self.seed = seed

        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.u_factors = self.init_params.get('U', None)
        self.i_factors = self.init_params.get('V', None)
        self.u_biases = self.init_params.get('Bu', None)
        self.i_biases = self.init_params.get('Bi', None)

    def _init(self):
        rng = get_rng(self.seed)

        if self.u_factors is None:
            self.u_factors = normal([self.num_users, self.k], std=0.01, random_state=rng) 
        if self.i_factors is None:
            self.i_factors = normal([self.num_items, self.k], std=0.01, random_state=rng)

        self.u_biases = zeros(self.num_users) if self.u_biases is None else self.u_biases
        self.i_biases = zeros(self.num_items) if self.i_biases is None else self.i_biases
        self.global_mean = self.global_mean if self.use_bias else 0.0

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

        if self.trainable:
            (rid, cid, val) = train_set.uir_tuple
            self._fit_sgd(rid, cid, val.astype(np.float32),
                          self.u_factors, self.i_factors, 
                          self.u_biases, self.i_biases)

        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_sgd(self, integral[:] rid, integral[:] cid, floating[:] val,
                 floating[:, :] U, floating[:, :] V, floating[:] Bu, floating[:] Bi):
        """Fit the model parameters (U, V, Bu, Bi) with SGD"""
        cdef:
            long num_users = self.num_users
            long num_items = self.num_items
            long num_ratings = val.shape[0]
            int num_factors = self.k
            int max_iter = self.max_iter
            int num_threads = self.num_threads

            floating reg = self.lambda_reg
            floating mu = self.global_mean

            bool use_bias = self.use_bias
            bool early_stop = self.early_stop
            bool verbose = self.verbose

            floating lr = self.learning_rate
            floating loss = 0
            floating last_loss = 0
            floating r, r_pred, error, u_f, i_f, delta_loss
            integral u, i, f, j

            floating * user
            floating * item

        progress = trange(max_iter, disable=not self.verbose)
        for epoch in progress:
            last_loss = loss
            loss = 0

            for j in prange(num_ratings, nogil=True, schedule='static', num_threads=num_threads):
                u, i, r = rid[j], cid[j], val[j]
                user, item = &U[u, 0], &V[i, 0]

                # predict rating
                r_pred = mu + Bu[u] + Bi[i]
                for f in range(num_factors):
                    r_pred = r_pred + user[f] * item[f]

                error = r - r_pred
                loss += error * error

                # update factors
                for f in range(num_factors):
                    u_f, i_f = user[f], item[f]
                    user[f] += lr * (error * i_f - reg * u_f)
                    item[f] += lr * (error * u_f - reg * i_f)

                # update biases
                if use_bias:
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
            known_item_scores = np.add(self.i_biases, self.global_mean)
            if self.knows_user(user_idx):
                known_item_scores = np.add(known_item_scores, self.u_biases[user_idx])
                fast_dot(self.u_factors[user_idx], self.i_factors, known_item_scores)
            return known_item_scores
        else:
            if self.use_bias:
                item_score = self.global_mean
                if self.knows_user(user_idx):
                    item_score += self.u_biases[user_idx]
                if self.knows_item(item_idx):
                    item_score += self.i_biases[item_idx]
                if self.knows_user(user_idx) and self.knows_item(item_idx):
                    item_score += np.dot(self.u_factors[user_idx], self.i_factors[item_idx])
            else:
                if not self.knows_user(user_idx) or not self.knows_item(item_idx):
                    raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_idx, item_idx))
                item_score = np.dot(self.u_factors[user_idx], self.i_factors[item_idx])
            return item_score

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
        user_vectors = self.u_factors
        if self.use_bias:
            user_vectors = np.concatenate(
                (
                    user_vectors,
                    self.u_biases.reshape((-1, 1)),
                    np.ones([user_vectors.shape[0], 1]), # augmented for item bias
                ), 
                axis=1
            )
        return user_vectors
    
    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model. 
        """
        item_vectors = self.i_factors
        if self.use_bias:
            item_vectors = np.concatenate(
                (
                    item_vectors,
                    np.ones([item_vectors.shape[0], 1]), # augmented for user bias
                    self.i_biases.reshape((-1, 1)),
                ), 
                axis=1
            )
        return item_vectors