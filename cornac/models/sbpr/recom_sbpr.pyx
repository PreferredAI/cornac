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
from cython cimport floating, integral
from cython.parallel import parallel, prange
from libc.math cimport exp, floor

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import fast_dot
from ...utils.common import scale
from ..bpr.recom_bpr cimport RNGVector, has_non_zero


cdef extern from "../bpr/recom_bpr.h" namespace "recom_bpr" nogil:
    cdef int get_thread_num()


class SBPR(Recommender):
    """Social Bayesian Personalized Ranking.

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
        Number of parallel threads for training.
        If 0, all CPU cores will be utilized.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors, 'Bi': item_biases}

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    * Zhao, T., McAuley, J., & King, I. (2014, November). Leveraging social connections to improve \
    personalized ranking for collaborative filtering. CIKM 2014 (pp. 261-270).
    """

    def __init__(self, name='SBPR', k=10, max_iter=100, learning_rate=0.001,
                 lambda_u=0.01, lambda_v=0.01, lambda_b=0.01,
                 num_threads=0, trainable=True, verbose=False, init_params=None, seed=None):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_b = lambda_b
        self.init_params = {} if init_params is None else init_params
        self.seed = seed

        import multiprocessing
        if num_threads > 0 and num_threads < multiprocessing.cpu_count():
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

        from tqdm import trange
        from ...utils import get_rng
        from ...utils.init_utils import zeros, uniform

        rng = get_rng(self.seed)
        self.u_factors = self.init_params.get('U', (uniform((n_users, self.k), random_state=rng) - 0.5) / self.k)
        self.i_factors = self.init_params.get('V', (uniform((n_items, self.k), random_state=rng) - 0.5) / self.k)
        self.i_biases = self.init_params.get('Bi', zeros(n_items))

        if not self.trainable:
            return

        # construct implicit feedback
        X = train_set.matrix # csr_matrix
        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(X.indptr)
        user_ids = np.repeat(np.arange(n_users), user_counts).astype(X.indices.dtype)

        # construct social feedback
        (rid, cid, val) = train_set.user_graph.get_train_triplet(train_set.uid_list,
                                                                     train_set.uid_list)
        Y = csr_matrix((val, (rid, cid)), shape=(n_users, n_users))
        social_item_ids = []
        social_item_counts = []
        social_indptr = [0]
        for uid in trange(n_users, disable=not self.verbose, desc='Social'):
            real_pos_items = np.unique(X[uid].indices)
            social_pos_items, counts = np.unique(X[Y[uid].indices].indices,
                                                     return_counts=True)
            mask = np.in1d(social_pos_items, real_pos_items, assume_unique=True)
            social_item_ids.extend(social_pos_items[~mask])
            social_item_counts.extend(counts[~mask])
            social_indptr.append(len(social_item_ids))

        social_item_ids = np.asarray(social_item_ids).astype(X.indices.dtype)
        social_item_counts = np.asarray(social_item_counts).astype(X.indices.dtype)
        social_indptr = np.asarray(social_indptr).astype(X.indices.dtype)

        # construct random generators
        cdef:
            int num_threads = self.num_threads
            RNGVector rng_pos = RNGVector(num_threads, len(user_ids) - 1)
            RNGVector rng_neg = RNGVector(num_threads, n_items - 1)

        # start training
        with trange(self.max_iter, disable=not self.verbose) as progress:
            for epoch in progress:
                skipped = self._fit_sgd(rng_pos, rng_neg, num_threads,
                                                 user_ids, X.indices, X.indptr,
                                                 social_item_ids, social_item_counts, social_indptr,
                                                 self.u_factors, self.i_factors, self.i_biases)
                progress.set_postfix({"skipped": "%.2f%%" % (100.0 * skipped / len(user_ids))})
        if self.verbose:
            print('Optimization finished!')

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_sgd(self, RNGVector rng_pos, RNGVector rng_neg, int num_threads,
                 integral[:] user_ids, integral[:] item_ids, integral[:] indptr,
                 integral[:] social_item_ids, integral[:] social_item_counts, integral[:] social_indptr,
                 floating[:, :] U, floating[:, :] V, floating[:] B):
        """Fit the model parameters (U, V, B) with SGD
        """
        cdef:
            long num_samples = len(user_ids)
            long num_items = self.train_set.num_items
            long s, i_index, k_index, skipped = 0
            int f, u_id, i_id, j_id, k_id, n_social_items, thread_id
            floating u_temp, k_rand
            floating z, score # for BPR formula
            floating z_ik, z_kj, score_ik, score_kj, s_uk # for SBPR-2 formula

            floating lr = self.learning_rate
            floating lbd_u = self.lambda_u
            floating lbd_v = self.lambda_v
            floating lbd_b = self.lambda_b
            int factors = self.k

            floating * user
            floating * item_i
            floating * item_j
            floating * item_k

        with nogil, parallel(num_threads=num_threads):
            thread_id = get_thread_num()

            for s in prange(num_samples, schedule='guided'):
                i_index = rng_pos.generate(thread_id)
                u_id = user_ids[i_index]
                i_id = item_ids[i_index]
                j_id = rng_neg.generate(thread_id)

                # sample social item k_id for given user u_id
                n_social_items = social_indptr[u_id + 1] - social_indptr[u_id]
                k_rand = <float>rng_neg.generate(thread_id) / num_items # uniform between [0.0, 1.0)
                k_index = social_indptr[u_id] + <int>floor(k_rand * n_social_items)
                k_id = social_item_ids[k_index]

                # if the user has liked the item j,
                # else if item j is also a social item,
                # skip this for now
                if has_non_zero(indptr, item_ids, u_id, j_id) or (j_id == k_id):
                    skipped += 1
                    continue

                # get pointers to the relevant factors
                user = &U[u_id, 0]
                item_i, item_j, item_k = &V[i_id, 0], &V[j_id, 0], &V[k_id, 0]

                # if no social item for given user uid, update factors based on BPR formula
                if n_social_items == 0:
                    # compute the score
                    score = B[i_id] - B[j_id]
                    for f in range(factors):
                        score = score + user[f] * (item_i[f] - item_j[f])
                    z = 1.0 / (1.0 + exp(score))

                    # update the factors via sgd.
                    for f in range(factors):
                        u_temp = user[f]
                        user[f] += lr * (z * (item_i[f] - item_j[f]) - lbd_u* user[f])
                        item_i[f] += lr * (z * u_temp - lbd_v * item_i[f])
                        item_j[f] += lr * (-z * u_temp - lbd_v * item_j[f])

                    # update item biases
                    B[i_id] += lr * (z - lbd_b * B[i_id])
                    B[j_id] += lr * (-z - lbd_b * B[j_id])

                    continue

                # found social feedback, update factors based on SBPR-2 formula
                # compute the scores
                score_ik = B[i_id] - B[k_id]
                score_kj = B[k_id] - B[j_id]
                for f in range(factors):
                    score_ik = score_ik + user[f] * (item_i[f] - item_k[f])
                    score_kj = score_kj + user[f] * (item_k[f] - item_j[f])
                s_uk = 1.0 / (1.0 + social_item_counts[k_index])
                z_ik = 1.0 / (1.0 + exp(score_ik * s_uk))
                z_kj = 1.0 / (1.0 + exp(score_kj))

                # update the factors via sgd.
                for f in range(factors):
                    u_temp = user[f]
                    user[f] += lr * (z_ik * (item_i[f] - item_k[f]) * s_uk +
                                         z_kj * (item_k[f] - item_j[f]) -
                                         lbd_u * user[f])
                    item_i[f] += lr * (z_ik * u_temp * s_uk - lbd_v * item_i[f])
                    item_j[f] += lr * (-z_kj * u_temp - lbd_v * item_j[f])
                    item_k[f] += lr * (z_kj * u_temp - z_ik * u_temp * s_uk - lbd_v * item_k[f])

                # update item biases
                B[i_id] += lr * (z_ik * s_uk - lbd_b * B[i_id])
                B[j_id] += lr * (-z_kj - lbd_b * B[j_id])
                B[k_id] += lr * (z_kj - z_ik * s_uk - lbd_b * B[k_id])

        return skipped

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
            known_item_scores = np.copy(self.i_biases)
            if not unk_user:
                fast_dot(self.u_factors[user_id], self.i_factors, known_item_scores)
            return known_item_scores
        else:
            if self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
            item_score = self.i_biases[item_id]
            if not unk_user:
                item_score += np.dot(self.u_factors[user_id], self.i_factors[item_id])
            if self.train_set.min_rating != self.train_set.max_rating:
                item_score = scale(item_score, self.train_set.min_rating, self.train_set.max_rating, 0., 1.)
            return item_score
