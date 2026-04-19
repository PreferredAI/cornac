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
import cython

cimport cython
from cython cimport floating, integral
from cython.parallel import parallel, prange
from libc.math cimport exp
from libcpp cimport bool
from libcpp.algorithm cimport binary_search
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np
import scipy.sparse as sp
from tqdm.auto import trange

from .recom_bpr cimport uniform_int_distribution, mt19937, has_non_zero, RNGVector
from ..recommender import Recommender
from ..recommender import ANNMixin, MEASURE_DOT
from ...exception import ScoreException
from ...utils import get_rng
from ...utils import fast_dot
from ...utils.common import scale
from ...utils.init_utils import uniform

DTYPE = np.float32

cdef extern from "recom_bpr.h" namespace "recom_bpr" nogil:
    cdef int get_thread_num()

@cython.boundscheck(False)

class VEBPR(Recommender, ANNMixin):
    """View-Enhanced Bayesian Personalized Ranking.

    This model extends BPR by incorporating users' view data as an intermediate
    feedback signal between purchases (positive) and unobserved items (negative).
    It jointly learns the pairwise rankings of user preference among these three
    types of interactions.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.01
        The learning rate for SGD.

    lambda_reg: float, optional, default: 0.1
        The regularization hyper-parameter.

    alpha: float, optional, default: 0.5
        The weight parameter controlling the relative strength between the two semantics
        of the view signal (negative compared to purchase, positive compared to unobserved).

    view_matrix: scipy.sparse matrix, required
        A user-item sparse matrix representing the view/click interactions.
        Must have the same shape as the training matrix.

    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Ding, J., Yu, G., He, X., Feng, F., Li, Y., & Jin, D. (2019). \
      Sampler Design for Bayesian Personalized Ranking by Leveraging View Data. \
      IEEE Transactions on Knowledge and Data Engineering.
    """

    def __init__(
            self,
            name='VEBPR',
            k=10,
            max_iter=100,
            learning_rate=0.01,
            lambda_reg=0.1,
            num_threads=0,
            trainable=True,
            verbose=False,
            init_params=None,
            seed=None,
            alpha=0.5,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = int(k)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.alpha = float(alpha)
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
        self.u_factor = self.init_params.get('U', None)
        self.i_factor = self.init_params.get('V', None)

    def _init(self):
        n_users, n_items = self.total_users, self.total_items

        if self.u_factor is None:
            self.u_factor = (uniform((n_users, self.k), random_state=self.rng, dtype=DTYPE) - 0.5) / self.k
        if self.i_factor is None:
            self.i_factor = (uniform((n_items, self.k), random_state=self.rng, dtype=DTYPE) - 0.5) / self.k

    def _prepare_data(self, train_set):
        X = train_set.matrix # csr_matrix
        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        V = getattr(train_set, 'view_matrix', None)
        # convert view matrix to CSR format
        if V is None:
            raise ValueError('VEBPR requires `view_matrix` to be provided')
        if not sp.isspmatrix(V):
            V = sp.csr_matrix(V)
        elif not sp.isspmatrix_csr(V):
            V = V.tocsr()
        # sort indices to ensure binary_search works correctly in c++
        V.sort_indices()
        # ensure purchase and view matrices share the same user-item shape
        if X.shape != V.shape:
            raise ValueError('`view_matrix` must have the same shape as train_set.matrix.')

        purchase_count = np.ediff1d(X.indptr).astype(np.int32)
        purchase_user_ids = np.repeat(np.arange(train_set.num_users), purchase_count).astype(X.indices.dtype)

        return X, V, purchase_count, purchase_user_ids

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection and early stopping (not strictly used).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if not hasattr(train_set, 'view_matrix'):
            raise ValueError("VEBPR requires a PurchaseViewDataset that contains a 'view_matrix' attribute.")

        self.view_matrix =  train_set.view_matrix
        self._init()

        if not self.trainable:
            return self

        X, V, purchase_count, purchase_user_ids = self._prepare_data(train_set)
        view_count = np.ediff1d(V.indptr).astype(np.int32)
        neg_item_ids = np.arange(train_set.num_items, dtype=np.int32)
        num_interactions = len(purchase_user_ids)

        cdef:
            int num_threads = self.num_threads
            RNGVector rng_pos = RNGVector(num_threads, num_interactions - 1, self.rng.randint(2 ** 31))
            RNGVector rng_view = RNGVector(num_threads, train_set.num_items - 1, self.rng.randint(2 ** 31))
            RNGVector rng_neg = RNGVector(num_threads, train_set.num_items - 1, self.rng.randint(2 ** 31))

        with trange(self.max_iter, disable=not self.verbose) as progress:
            for _ in progress:
                correct, skipped = self._fit_sgd_viewloss(
                    rng_pos, rng_view, rng_neg, num_threads, purchase_count, view_count, purchase_user_ids, X.indices,
                    X.indptr,
                    V.indices, V.indptr, neg_item_ids, self.u_factor, self.i_factor
                )
                progress.set_postfix({
                    'correct': '%.2f%%' % (100.0 * correct / (num_interactions - skipped + 1e-8)),
                    'skipped': '%.2f%%' % (100.0 * skipped / num_interactions)
                })

        if self.verbose:
            print('Optimization finished!')
        return self

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_sgd_viewloss(
            self, RNGVector rng_pos, RNGVector rng_view, RNGVector rng_neg, int num_threads,
            integral[:] purchase_count, integral[:] view_count, integral[:] purchase_user_ids,
            integral[:] purchase_item_ids,
            integral[:] purchase_indptr, integral[:] view_item_ids, integral[:] view_indptr, integral[:] neg_item_ids,
            floating[:, :] U, floating[:, :] V
    ):
        """Internal Cython optimization loop for VEBPR."""
        cdef:
            long num_samples = len(purchase_user_ids), s, i_index, v_index, j_index, correct = 0, skipped = 0
            integral f, u_id, i_id, v_id, j_id, thread_id
            integral num_purchase, num_view
            floating delta_ij, delta_iv, delta_vj, x_uij, x_uiv, x_uvj
            floating u_old, i_old, v_old, j_old
            floating alpha = self.alpha
            floating lr = self.learning_rate
            floating reg = self.lambda_reg
            int factor = self.k
            floating * user
            floating * item_i
            floating * item_v
            floating * item_j

        with nogil, parallel(num_threads=num_threads):
            thread_id = get_thread_num()
            for s in prange(num_samples, schedule='guided'):
                i_index = rng_pos.generate(thread_id) % num_samples
                u_id = purchase_user_ids[i_index]
                i_id = purchase_item_ids[i_index]
                num_purchase = purchase_count[u_id]
                num_view = view_count[u_id]

                if num_purchase == 0 or num_view == 0:
                    skipped += 1
                    continue

                v_index = view_indptr[u_id] + (rng_view.generate(thread_id) % num_view)
                v_id = view_item_ids[v_index]
                j_index = rng_neg.generate(thread_id)
                j_id = neg_item_ids[j_index]
                if has_non_zero(purchase_indptr, purchase_item_ids, u_id, j_id) or has_non_zero(view_indptr,
                                                                                                view_item_ids, u_id,
                                                                                                j_id):
                    skipped += 1
                    continue
                # get pointers to the relevant factors
                user = &U[u_id, 0]
                item_i = &V[i_id, 0]
                item_v = &V[v_id, 0]
                item_j = &V[j_id, 0]

                x_uij = 0.0
                x_uiv = 0.0
                x_uvj = 0.0
                for f in range(factor):
                    x_uij = x_uij + user[f] * (item_i[f] - item_j[f])
                    x_uiv = x_uiv + user[f] * (item_i[f] - item_v[f])
                    x_uvj = x_uvj + user[f] * (item_v[f] - item_j[f])

                if x_uij > 50.0:
                    x_uij = 50.0
                elif x_uij < -50.0:
                    x_uij = -50.0

                if x_uiv > 50.0:
                    x_uiv = 50.0
                elif x_uiv < -50.0:
                    x_uiv = -50.0

                if x_uvj > 50.0:
                    x_uvj = 50.0
                elif x_uvj < -50.0:
                    x_uvj = -50.0

                delta_ij = 1.0 / (1.0 + exp(x_uij))
                delta_iv = 1.0 / (1.0 + exp(x_uiv))
                delta_vj = 1.0 / (1.0 + exp(x_uvj))

                if delta_ij < 0.5 and delta_iv < 0.5 and delta_vj < 0.5:
                    correct += 1
                # update the factors via sgd.
                for f in range(factor):
                    u_old = user[f]
                    i_old = item_i[f]
                    v_old = item_v[f]
                    j_old = item_j[f]

                    user[f] -= lr * (
                            - delta_ij * (i_old - j_old)
                            - alpha * delta_iv * (i_old - v_old)
                            - (1.0 - alpha) * delta_vj * (v_old - j_old)
                            + reg * u_old
                    )
                    item_i[f] -= lr * (-delta_ij * u_old - alpha * delta_iv * u_old + reg * i_old)
                    item_v[f] -= lr * (alpha * delta_iv * u_old - (1.0 - alpha) * delta_vj * u_old + reg * v_old)
                    item_j[f] -= lr * (delta_ij * u_old + (1.0 - alpha) * delta_vj * u_old + reg * j_old)

        return correct, skipped

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for all items.

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
            know_item_scores = np.zeros(self.i_factor.shape[0], dtype=DTYPE)
            fast_dot(self.u_factor[user_idx], self.i_factor, know_item_scores)
            return know_item_scores
        else:
            score = np.dot(self.u_factor[user_idx], self.i_factor[item_idx])
            return score

    def get_vector_measure(self):
        """Getting a valid choice of vector measurement in ANNMixin._ann_setup.

        Returns
        -------
        measure : MEASURE_DOT
            Dot product aka. inner product
        """
        return MEASURE_DOT

    def get_user_vectors(self):
        """Getting a matrix of user vectors serving as query for ANN search.

        Returns
        -------
        out : numpy.array
            Matrix of user vectors for all users
        """
        return self.u_factor

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out : numpy.array
            Matrix of item vectors for all items
        """
        return self.i_factor