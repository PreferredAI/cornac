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

import numpy as np
cimport numpy as np

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import fast_dot


class NMF(Recommender):
    """Non-negative Matrix Factorization

    Parameters
    ----------
    k: int, optional, default: 15
        The dimension of the latent factors.

    max_iter: int, optional, default: 50
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.005
        The learning rate.

    lambda_u: float, optional, default: 0.06
        The regularization parameter for user factors U.

    lambda_v: float, optional, default: 0.06
        The regularization parameter for item factors V.

    lambda_bu: float, optional, default: 0.02
        The regularization parameter for user biases Bu.

    lambda_bi: float, optional, default: 0.02
        The regularization parameter for item biases Bi.

    use_bias: boolean, optional, default: False
        When True, user, item, and global biases are used.

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors,
        'Bu': user_biases, 'Bi': item_biases, 'mu': global_mean}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. \
    In Advances in neural information processing systems (pp. 556-562).

    * Takahashi, N., Katayama, J., & Takeuchi, J. I. (2014). A generalized sufficient condition for \
    global convergence of modified multiplicative updates for NMF. \
    In Proceedings of 2014 International Symposium on Nonlinear Theory and its Applications (pp. 44-47).
    """

    def __init__(self, name='NMF', k=15, max_iter=50, learning_rate=.005,
                 lambda_u=.06, lambda_v=.06, lambda_bu=.02, lambda_bi=.02,
                 use_bias=False, num_threads=0,
                 trainable=True, verbose=False, init_params=None, seed=None):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_bu = lambda_bu
        self.lambda_bi = lambda_bi
        self.use_bias = use_bias
        self.init_params = {} if init_params is None else init_params
        self.seed = seed
        import multiprocessing
        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

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

        from ...utils import get_rng
        from ...utils.init_utils import uniform, zeros

        rng = get_rng(self.seed)
        n_users, n_items = train_set.num_users, train_set.num_items
        self.u_factors = self.init_params.get('U', uniform((n_users, self.k), random_state=rng))
        self.i_factors = self.init_params.get('V', uniform((n_items, self.k), random_state=rng))
        self.u_biases = self.init_params.get('Bu', zeros(n_users))
        self.i_biases = self.init_params.get('Bi', zeros(n_items))
        self.global_mean = self.init_params.get('mu', train_set.global_mean) if self.use_bias else 0.

        if self.trainable:
            X = train_set.matrix # csr_matrix
            user_counts = np.ediff1d(X.indptr)
            user_ids = np.repeat(np.arange(n_users), user_counts).astype(X.indices.dtype)
            item_counts = np.ediff1d(X.tocsc().indptr).astype(X.indices.dtype)

            self._fit_sgd(user_ids, X.indices, X.data.astype(np.float32), user_counts, item_counts,
                          self.u_factors, self.i_factors, self.u_biases, self.i_biases)

        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_sgd(self, integral[:] rid, integral[:] cid, floating[:] val,
                 integral[:] user_counts, integral[:] item_counts,
                 floating[:, :] U, floating[:, :] V, floating[:] Bu, floating[:] Bi):
        """Fit the model parameters (U, V, Bu, Bi)
        """
        cdef:
            long num_users = self.train_set.num_users
            long num_items = self.train_set.num_items
            long num_ratings = val.shape[0]
            int num_factors = self.k
            int max_iter = self.max_iter
            int num_threads = self.num_threads
            floating lr = self.learning_rate

            floating lambda_u = self.lambda_u
            floating lambda_v = self.lambda_v
            floating lambda_bu = self.lambda_bu
            floating lambda_bi = self.lambda_bi
            floating mu = self.global_mean

            bool use_bias = self.use_bias
            bool verbose = self.verbose

            np.ndarray[np.float32_t, ndim=2] U_numerator = np.empty((num_users, num_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] V_numerator = np.empty((num_items, num_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] U_denominator = np.empty((num_users, num_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] V_denominator = np.empty((num_items, num_factors), dtype=np.float32)

            floating loss, r, r_pred, error, eps = 1e-9

            long u, i, f, j

        from tqdm import trange
        progress = trange(max_iter, disable=not verbose)
        for epoch in progress:
            loss = 0.
            U_numerator.fill(0)
            V_numerator.fill(0)
            U_denominator.fill(0)
            V_denominator.fill(0)

            for j in prange(num_ratings, nogil=True, num_threads=num_threads):
                u, i, r = rid[j], cid[j], val[j]

                # predict rating
                r_pred = mu + Bu[u] + Bi[i]
                for f in range(num_factors):
                    r_pred = r_pred + U[u, f] * V[i, f]

                error = r - r_pred
                loss += error * error

                # update biases
                if use_bias:
                    Bu[u] += lr * (error - lambda_bu * Bu[u])
                    Bi[i] += lr * (error - lambda_bi * Bi[i])

                # compute numerators and denominators
                for f in range(num_factors):
                    U_numerator[u, f] += r * V[i, f]
                    U_denominator[u, f] += r_pred * V[i, f]
                    V_numerator[i, f] += r * U[u, f]
                    V_denominator[i, f] += r_pred * U[u, f]

            # update user factors
            for u in prange(num_users, nogil=True, num_threads=num_threads):
                for f in range(num_factors):
                    loss += lambda_u * U[u, f] * U[u, f]
                    U_denominator[u, f] += user_counts[u] * lambda_u * U[u, f] + eps
                    U[u, f] *= U_numerator[u, f] / U_denominator[u, f]

            # update item factors
            for i in prange(num_items, nogil=True, num_threads=num_threads):
                for f in range(num_factors):
                    loss += lambda_v * V[i, f] * V[i, f]
                    V_denominator[i, f] += item_counts[i] * lambda_v * V[i, f] + eps
                    V[i, f] *= V_numerator[i, f] / V_denominator[i, f]

            progress.set_postfix({"loss": "%.2f" % loss})
            progress.update(1)

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
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        unk_user = self.train_set.is_unk_user(user_idx)

        if item_idx is None:
            known_item_scores = np.add(self.i_biases, self.global_mean)
            if not unk_user:
                known_item_scores = np.add(known_item_scores, self.u_biases[user_idx])
                fast_dot(self.u_factors[user_idx], self.i_factors, known_item_scores)
            return known_item_scores
        else:
            unk_item = self.train_set.is_unk_item(item_idx)
            if self.use_bias:
                item_score = self.global_mean
                if not unk_user:
                    item_score += self.u_biases[user_idx]
                if not unk_item:
                    item_score += self.i_biases[item_idx]
                if not unk_user and not unk_item:
                    item_score += np.dot(self.u_factors[user_idx], self.i_factors[item_idx])
            else:
                if unk_user or unk_item:
                    raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_idx, item_idx))
                item_score = np.dot(self.u_factors[user_idx], self.i_factors[item_idx])
            return item_score
