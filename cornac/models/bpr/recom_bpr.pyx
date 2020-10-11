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
from cython cimport floating, integral
from cython.parallel import parallel, prange
from libc.math cimport exp
from libcpp cimport bool
from libcpp.algorithm cimport binary_search

import numpy as np
cimport numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils import fast_dot
from ...utils.common import scale
from ...utils.init_utils import zeros, uniform


cdef extern from "recom_bpr.h" namespace "recom_bpr" nogil:
    cdef int get_thread_num()


@cython.boundscheck(False)
cdef bool has_non_zero(integral[:] indptr, integral[:] indices,
                       integral rowid, integral colid) nogil:
    """Given a CSR matrix, returns whether the [rowid, colid] contains a non zero.
    Assumes the CSR matrix has sorted indices"""
    return binary_search(&indices[indptr[rowid]], &indices[indptr[rowid + 1]], colid)


cdef class RNGVector(object):
    def __init__(self, int num_threads, long rows, int seed):
        rng = get_rng(seed)
        for i in range(num_threads):
            self.rng.push_back(mt19937(rng.randint(2 ** 31)))
            self.dist.push_back(uniform_int_distribution[long](0, rows))

    cdef inline long generate(self, int thread_id) nogil:
        return self.dist[thread_id](self.rng[thread_id])



class BPR(Recommender):
    """Bayesian Personalized Ranking.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    lambda_reg: float, optional, default: 0.001
        The regularization hyper-parameter.

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors, 'Bi': item_biases}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Rendle, Steffen, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. \
    BPR: Bayesian personalized ranking from implicit feedback. In UAI, pp. 452-461. 2009.
    """

    def __init__(
        self, 
        name='BPR', 
        k=10, 
        max_iter=100, 
        learning_rate=0.001, 
        lambda_reg=0.01,
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
        self.u_factors = self.init_params.get('U', None)
        self.i_factors = self.init_params.get('V', None)
        self.i_biases = self.init_params.get('Bi', None)
    
    def _init(self):
        n_users, n_items = self.train_set.total_users, self.train_set.total_items

        if self.u_factors is None:
            self.u_factors = (uniform((n_users, self.k), random_state=self.rng) - 0.5) / self.k
        if self.i_factors is None:
            self.i_factors = (uniform((n_items, self.k), random_state=self.rng) - 0.5) / self.k
        self.i_biases = zeros(n_items) if self.i_biases is None else self.i_biases

    def _prepare_data(self):
        X = self.train_set.matrix # csr_matrix
        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(X.indptr)
        user_ids = np.repeat(np.arange(self.train_set.num_users), user_counts).astype(X.indices.dtype)

        return X, user_counts, user_ids

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

        X, user_counts, user_ids = self._prepare_data()
        neg_item_ids = np.arange(train_set.num_items, dtype=np.int32)

        cdef:
            int num_threads = self.num_threads
            RNGVector rng_pos = RNGVector(num_threads, len(user_ids) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_neg = RNGVector(num_threads, train_set.num_items - 1, self.rng.randint(2 ** 31))

        with trange(self.max_iter, disable=not self.verbose) as progress:
            for epoch in progress:
                correct, skipped = self._fit_sgd(rng_pos, rng_neg, num_threads,
                                                user_ids, X.indices, neg_item_ids, X.indptr,
                                                self.u_factors, self.i_factors, self.i_biases)
                progress.set_postfix({
                    "correct": "%.2f%%" % (100.0 * correct / (len(user_ids) - skipped)),
                    "skipped": "%.2f%%" % (100.0 * skipped / len(user_ids))
                })

        if self.verbose:
            print('Optimization finished!')

        return self

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_sgd(self, RNGVector rng_pos, RNGVector rng_neg, int num_threads,
                 integral[:] user_ids, integral[:] item_ids, 
                 integral[:] neg_item_ids, integral[:] indptr,
                 floating[:, :] U, floating[:, :] V, floating[:] B):
        """Fit the model parameters (U, V, B) with SGD
        """
        cdef:
            long num_samples = len(user_ids), s, i_index, j_index, correct = 0, skipped = 0
            long num_items = self.train_set.num_items
            integral f, i_id, j_id, thread_id
            floating z, score, temp

            floating lr = self.learning_rate
            floating reg = self.lambda_reg
            int factors = self.k

            floating * user
            floating * item_i
            floating * item_j

        with nogil, parallel(num_threads=num_threads):
            thread_id = get_thread_num()

            for s in prange(num_samples, schedule='guided'):
                i_index = rng_pos.generate(thread_id)
                i_id = item_ids[i_index]
                j_index = rng_neg.generate(thread_id)
                j_id = neg_item_ids[j_index]

                # if the user has liked the item j, skip this for now
                if has_non_zero(indptr, item_ids, user_ids[i_index], j_id):
                    skipped += 1
                    continue

                # get pointers to the relevant factors
                user, item_i, item_j = &U[user_ids[i_index], 0], &V[i_id, 0], &V[j_id, 0]

                # compute the score
                score = B[i_id] - B[j_id]
                for f in range(factors):
                    score = score + user[f] * (item_i[f] - item_j[f])
                z = 1.0 / (1.0 + exp(score))

                if z < .5:
                    correct += 1

                # update the factors via sgd.
                for f in range(factors):
                    temp = user[f]
                    user[f] += lr * (z * (item_i[f] - item_j[f]) - reg * user[f])
                    item_i[f] += lr * (z * temp - reg * item_i[f])
                    item_j[f] += lr * (-z * temp - reg * item_j[f])

                # update item biases
                B[i_id] += lr * (z - reg * B[i_id])
                B[j_id] += lr * (-z - reg * B[j_id])

        return correct, skipped


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
            known_item_scores = np.copy(self.i_biases)
            fast_dot(self.u_factors[user_idx], self.i_factors, known_item_scores)
            return known_item_scores
        else:
            item_score = self.i_biases[item_idx]
            item_score += np.dot(self.u_factors[user_idx], self.i_factors[item_idx])
            return item_score