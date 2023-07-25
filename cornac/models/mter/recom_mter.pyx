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

import multiprocessing as mp

cimport cython
from cython cimport floating, integral
from cython.parallel import parallel, prange
from libc.math cimport sqrt, log, exp

import scipy.sparse as sp
import numpy as np
cimport numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.fast_dict cimport IntFloatDict
from ...utils.init_utils import uniform
from ..bpr.recom_bpr cimport RNGVector, has_non_zero


cdef extern from "../bpr/recom_bpr.h" namespace "recom_bpr" nogil:
    cdef int get_thread_num()


cdef int get_key(int i_id, int j_id) nogil:
    return (i_id + j_id) * (i_id + j_id + 1) // 2 + j_id


@cython.boundscheck(False)
@cython.wraparound(False)
cdef floating get_score(floating[:, :, :] G, int dim1, int dim2, int dim3,
                        floating[:, :] U, floating[:, :] I, floating[:, :] A,
                        int u_idx, int i_idx, int a_idx) nogil:
    cdef floating score = 0.
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                score = score + G[i, j, k] * U[u_idx, i] * I[i_idx, j] * A[a_idx, k]
    return score


class MTER(Recommender):
    """Multi-Task Explainable Recommendation

    Parameters
    ----------
    name: string, optional, default: 'MTER'
        The name of the recommender model.

    rating_scale: float, optional, default: 5.0
        The maximum rating score of the dataset.

    n_user_factors: int, optional, default: 15
        The dimension of the user latent factors.

    n_item_factors: int, optional, default: 15
        The dimension of the item latent factors.

    n_aspect_factors: int, optional, default: 12
        The dimension of the aspect latent factors.

    n_opinion_factors: int, optional, default: 12
        The dimension of the opinion latent factors.

    n_bpr_samples: int, optional, default: 1000
        The number of samples from all BPR pairs.

    n_element_samples: int, optional, default: 50
        The number of samples from all ratings in each iteration.

    lambda_reg: float, optional, default: 0.1
        The regularization parameter.

    lambda_bpr: float, optional, default: 10.0
        The regularization parameter for BPR.

    max_iter: int, optional, default: 200000
        Maximum number of iterations for training.

    lr: float, optional, default: 0.1
        The learning rate for optimization

    n_threads: int, optional, default: 0
        Number of parallel threads for training. If n_threads=0, all CPU cores will be utilized.
        If seed is not None, n_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U, I, A, O, G1, G2, and G3 are not None).

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'I':I, 'A':A, 'O':O, 'G1':G1, 'G2':G2, 'G3':G3}

        U: ndarray, shape (n_users, n_user_factors)
            The user latent factors, optional initialization via init_params
            
        I: ndarray, shape (n_items, n_item_factors)
            The item latent factors, optional initialization via init_params
        
        A: ndarray, shape (num_aspects+1, n_aspect_factors)
            The aspect latent factors, optional initialization via init_params
        
        O: ndarray, shape (num_opinions, n_opinion_factors)
            The opinion latent factors, optional initialization via init_params

        G1: ndarray, shape (n_user_factors, n_item_factors, n_aspect_factors)
            The core tensor for user, item, and aspect factors, optional initialization via init_params

        G2: ndarray, shape (n_user_factors, n_aspect_factors, n_opinion_factors)
            The core tensor for user, aspect, and opinion factors, optional initialization via init_params

        G3: ndarray, shape (n_item_factors, n_aspect_factors, n_opinion_factors)
            The core tensor for item, aspect, and opinion factors, optional initialization via init_params

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    Nan Wang, Hongning Wang, Yiling Jia, and Yue Yin. 2018. \
    Explainable Recommendation via Multi-Task Learning in Opinionated Text Data. \
    In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval (SIGIR '18). \
    ACM, New York, NY, USA, 165-174. DOI: https://doi.org/10.1145/3209978.3210010
    """
    def __init__(
        self,
        name="MTER",
        rating_scale=5.0,
        n_user_factors=15,
        n_item_factors=15,
        n_aspect_factors=12,
        n_opinion_factors=12,
        n_bpr_samples=1000,
        n_element_samples=50,
        lambda_reg=0.1,
        lambda_bpr=10,
        max_iter=200000,
        lr=0.1,
        n_threads=0,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.rating_scale = rating_scale
        self.n_user_factors = n_user_factors
        self.n_item_factors = n_item_factors
        self.n_aspect_factors = n_aspect_factors
        self.n_opinion_factors = n_opinion_factors
        self.n_bpr_samples = n_bpr_samples
        self.n_element_samples = n_element_samples
        self.lambda_reg = lambda_reg
        self.lambda_bpr = lambda_bpr
        self.max_iter = max_iter
        self.lr = lr
        self.seed = seed

        if seed is not None:
            self.n_threads = 1
        elif n_threads > 0 and n_threads < mp.cpu_count():
            self.n_threads = n_threads
        else:
            self.n_threads = mp.cpu_count()
        self.rng = get_rng(seed)

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.G1 = self.init_params.get("G1", None)
        self.G2 = self.init_params.get("G2", None)
        self.G3 = self.init_params.get("G3", None)
        self.U = self.init_params.get("U", None)
        self.I = self.init_params.get("I", None)
        self.A = self.init_params.get("A", None)
        self.O = self.init_params.get("O", None)

    def _init(self):
        n_users, n_items = self.train_set.num_users, self.train_set.num_items
        n_aspects, n_opinions = self.train_set.sentiment.num_aspects, self.train_set.sentiment.num_opinions

        if self.G1 is None:
            G1_shape = (self.n_user_factors, self.n_item_factors, self.n_aspect_factors)
            self.G1 = uniform(G1_shape, random_state=self.rng)
        if self.G2 is None:
            G2_shape = (self.n_user_factors, self.n_aspect_factors, self.n_opinion_factors)
            self.G2 = uniform(G2_shape, random_state=self.rng)
        if self.G3 is None:
            G3_shape = (self.n_item_factors, self.n_aspect_factors, self.n_opinion_factors)
            self.G3 = uniform(G3_shape, random_state=self.rng)
        if self.U is None:
            U_shape = (n_users, self.n_user_factors)
            self.U = uniform(U_shape, random_state=self.rng)
        if self.I is None:
            I_shape = (n_items, self.n_item_factors)
            self.I = uniform(I_shape, random_state=self.rng)
        if self.A is None:
            A_shape = (n_aspects + 1, self.n_aspect_factors)
            self.A = uniform(A_shape, random_state=self.rng)
        if self.O is None:
            O_shape = (n_opinions, self.n_opinion_factors)
            self.O = uniform(O_shape, random_state=self.rng)

    def _build_data(self, data_set):
        import time

        start_time = time.time()
        if self.verbose:
            print("Building data started!")

        sentiment = self.train_set.sentiment
        (u_indices, i_indices, r_values) = data_set.uir_tuple
        keys = np.array([get_key(u, i) for u, i in zip(u_indices, i_indices)], dtype=np.intp)
        cdef IntFloatDict rating_dict = IntFloatDict(keys, np.array(r_values, dtype=np.float64))
        rating_matrix = sp.csr_matrix(
            (r_values, (u_indices, i_indices)),
            shape=(self.train_set.num_users, self.train_set.num_items),
        )
        user_item_aspect = {}
        user_aspect_opinion = {}
        item_aspect_opinion = {}
        for u_idx, sentiment_tup_ids_by_item in sentiment.user_sentiment.items():
            if self.train_set.is_unk_user(u_idx):
                continue
            for i_idx, tup_idx in sentiment_tup_ids_by_item.items():
                user_item_aspect[
                    (u_idx, i_idx, sentiment.num_aspects)
                ] = rating_matrix[u_idx, i_idx]
                for a_idx, o_idx, polarity in sentiment.sentiment[tup_idx]:
                    user_item_aspect[(u_idx, i_idx, a_idx)] = (
                        user_item_aspect.get((u_idx, i_idx, a_idx), 0)
                        + polarity
                    )
                    if (
                        polarity > 0
                    ):  # only include opinion with positive sentiment polarity
                        user_aspect_opinion[(u_idx, a_idx, o_idx)] = (
                            user_aspect_opinion.get(
                                (u_idx, a_idx, o_idx), 0
                            )
                            + 1
                        )
                        item_aspect_opinion[(i_idx, a_idx, o_idx)] = (
                            item_aspect_opinion.get(
                                (i_idx, a_idx, o_idx), 0
                            )
                            + 1
                        )

        for key in user_item_aspect.keys():
            if key[2] != sentiment.num_aspects:
                user_item_aspect[key] = self._compute_quality_score(
                    user_item_aspect[key]
                )

        for key in user_aspect_opinion.keys():
            user_aspect_opinion[key] = self._compute_attention_score(
                user_aspect_opinion[key]
            )

        for key in item_aspect_opinion.keys():
            item_aspect_opinion[key] = self._compute_attention_score(
                item_aspect_opinion[key]
            )

        if self.verbose:
            total_time = time.time() - start_time
            print("Building data completed in %d s" % total_time)
        return (
            rating_matrix,
            rating_dict,
            user_item_aspect,
            user_aspect_opinion,
            item_aspect_opinion,
        )

    def _compute_attention_score(self, count):
        return 1 + (self.rating_scale - 1) * (2 / (1 + np.exp(-count)) - 1)

    def _compute_quality_score(self, sentiment):
        return 1 + (self.rating_scale - 1) / (1 + np.exp(-sentiment))

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

        (
            rating_matrix,
            rating_dict,
            user_item_aspect,
            user_aspect_opinion,
            item_aspect_opinion,
        ) = self._build_data(train_set)

        X, X_uids, X_iids, X_aids = [], [], [], []
        YU, YU_uids, YU_aids, YU_oids = [], [], [], []
        YI, YI_iids, YI_aids, YI_oids = [], [], [], []

        for (uid, iid, aid), score in user_item_aspect.items():
            X.append(score)
            X_uids.append(uid)
            X_iids.append(iid)
            X_aids.append(aid)

        for (uid, aid, oid), score in user_aspect_opinion.items():
            YU.append(score)
            YU_uids.append(uid)
            YU_aids.append(aid)
            YU_oids.append(oid)

        for (iid, aid, oid), score in item_aspect_opinion.items():
            YI.append(score)
            YI_iids.append(iid)
            YI_aids.append(aid)
            YI_oids.append(oid)

        X = np.array(X, dtype=np.float32)
        X_uids = np.array(X_uids, dtype=np.int32)
        X_iids = np.array(X_iids, dtype=np.int32)
        X_aids = np.array(X_aids, dtype=np.int32)
        YU = np.array(YU, dtype=np.float32)
        YU_uids = np.array(YU_uids, dtype=np.int32)
        YU_aids = np.array(YU_aids, dtype=np.int32)
        YU_oids = np.array(YU_oids, dtype=np.int32)
        YI = np.array(YI, dtype=np.float32)
        YI_iids = np.array(YI_iids, dtype=np.int32)
        YI_aids = np.array(YI_aids, dtype=np.int32)
        YI_oids = np.array(YI_oids, dtype=np.int32)

        user_counts = np.ediff1d(rating_matrix.indptr).astype(np.int32)
        user_ids = np.repeat(np.arange(self.train_set.num_users), user_counts).astype(np.int32)
        neg_item_ids = np.arange(train_set.num_items, dtype=np.int32)

        cdef:
            int n_threads = self.n_threads
            RNGVector rng_pos_uia = RNGVector(n_threads, len(user_item_aspect) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_pos_uao = RNGVector(n_threads, len(user_aspect_opinion) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_pos_iao = RNGVector(n_threads, len(item_aspect_opinion) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_pos = RNGVector(n_threads, len(user_ids) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_neg = RNGVector(n_threads, train_set.num_items - 1, self.rng.randint(2 ** 31))

        sgrad_G1 = np.zeros_like(self.G1).astype(np.float32)
        sgrad_G2 = np.zeros_like(self.G2).astype(np.float32)
        sgrad_G3 = np.zeros_like(self.G3).astype(np.float32)
        sgrad_U = np.zeros_like(self.U).astype(np.float32)
        sgrad_I = np.zeros_like(self.I).astype(np.float32)
        sgrad_A = np.zeros_like(self.A).astype(np.float32)
        sgrad_O = np.zeros_like(self.O).astype(np.float32)
        del_g1 = np.zeros_like(self.G1).astype(np.float32)
        del_g2 = np.zeros_like(self.G2).astype(np.float32)
        del_g3 = np.zeros_like(self.G3).astype(np.float32)
        del_u = np.zeros_like(self.U).astype(np.float32)
        del_i = np.zeros_like(self.I).astype(np.float32)
        del_a = np.zeros_like(self.A).astype(np.float32)
        del_o = np.zeros_like(self.O).astype(np.float32)
        del_g1_reg = np.zeros_like(self.G1).astype(np.float32)
        del_g2_reg = np.zeros_like(self.G2).astype(np.float32)
        del_g3_reg = np.zeros_like(self.G3).astype(np.float32)
        del_u_reg = np.zeros_like(self.U).astype(np.float32)
        del_i_reg = np.zeros_like(self.I).astype(np.float32)
        del_a_reg = np.zeros_like(self.A).astype(np.float32)
        del_o_reg = np.zeros_like(self.O).astype(np.float32)

        with trange(self.max_iter, disable=not self.verbose) as progress:
            for epoch in progress:
                correct, skipped, loss, bpr_loss = self._fit_mter(
                    rng_pos_uia, rng_pos_uao, rng_pos_iao, rng_pos, rng_neg,
                    n_threads,
                    X, X_uids, X_iids, X_aids,
                    YU, YU_uids, YU_aids, YU_oids,
                    YI, YI_iids, YI_aids, YI_oids,
                    rating_dict,
                    user_ids, rating_matrix.indices, neg_item_ids, rating_matrix.indptr,
                    self.G1, self.G2, self.G3, self.U, self.I, self.A, self.O,
                    sgrad_G1, sgrad_G2, sgrad_G3, sgrad_U, sgrad_I, sgrad_A, sgrad_O,
                    del_g1, del_g2, del_g3, del_u, del_i, del_a, del_o,
                    del_g1_reg, del_g2_reg, del_g3_reg, del_u_reg, del_i_reg, del_a_reg, del_o_reg
                )

                progress.set_postfix({
                    "loss": "%.2f" % (loss / 3 / self.n_element_samples),
                    "bpr_loss": "%.2f" % (bpr_loss / self.n_bpr_samples),
                    "correct": "%.2f%%" % (100.0 * correct / (self.n_bpr_samples - skipped)),
                    "skipped": "%.2f%%" % (100.0 * skipped / self.n_bpr_samples)
                })

        if self.verbose:
            print('Optimization finished!')

        return self

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_mter(
        self,
        RNGVector rng_pos_uia,
        RNGVector rng_pos_uao,
        RNGVector rng_pos_iao,
        RNGVector rng_pos,
        RNGVector rng_neg,
        int n_threads,
        floating[:] X, integral[:] X_uids, integral[:] X_iids, integral[:] X_aids,
        floating[:] YU, integral[:] YU_uids, integral[:] YU_aids, integral[:] YU_oids,
        floating[:] YI, integral[:] YI_iids, integral[:] YI_aids, integral[:] YI_oids,
        IntFloatDict rating_dict,
        integral[:] user_ids, integral[:] item_ids, integral[:] neg_item_ids, integral[:] indptr,
        floating[:, :, :] G1,
        floating[:, :, :] G2,
        floating[:, :, :] G3,
        floating[:, :] U,
        floating[:, :] I,
        floating[:, :] A,
        floating[:, :] O,
        floating[:, :, :] sgrad_G1,
        floating[:, :, :] sgrad_G2,
        floating[:, :, :] sgrad_G3,
        floating[:, :] sgrad_U,
        floating[:, :] sgrad_I,
        floating[:, :] sgrad_A,
        floating[:, :] sgrad_O,
        np.ndarray[np.float32_t, ndim=3] del_g1,
        np.ndarray[np.float32_t, ndim=3] del_g2,
        np.ndarray[np.float32_t, ndim=3] del_g3,
        np.ndarray[np.float32_t, ndim=2] del_u,
        np.ndarray[np.float32_t, ndim=2] del_i,
        np.ndarray[np.float32_t, ndim=2] del_a,
        np.ndarray[np.float32_t, ndim=2] del_o,
        np.ndarray[np.float32_t, ndim=3] del_g1_reg,
        np.ndarray[np.float32_t, ndim=3] del_g2_reg,
        np.ndarray[np.float32_t, ndim=3] del_g3_reg,
        np.ndarray[np.float32_t, ndim=2] del_u_reg,
        np.ndarray[np.float32_t, ndim=2] del_i_reg,
        np.ndarray[np.float32_t, ndim=2] del_a_reg,
        np.ndarray[np.float32_t, ndim=2] del_o_reg):
        """Fit the model parameters (G1, G2, G3, U, I, A, O)
        """
        cdef:
            long s, i_index, j_index, correct = 0, skipped = 0
            long n_users = self.train_set.num_users
            long n_items = self.train_set.num_items
            long n_aspects = self.train_set.sentiment.num_aspects
            long n_opinions = self.train_set.sentiment.num_opinions
            long n_user_factors = self.n_user_factors
            long n_item_factors = self.n_item_factors
            long n_aspect_factors = self.n_aspect_factors
            long n_opinion_factors = self.n_opinion_factors
            int num_samples = self.n_element_samples
            int num_bpr_samples = self.n_bpr_samples

            integral _, i, j, k, idx, jdx, u_idx, i_idx, a_idx, o_idx, j_idx, thread_id
            floating z, score, i_score, j_score, pred, temp, i_ij
            floating loss = 0., bpr_loss = 0., del_sqerror, del_bpr
            floating eps = 1e-9

            floating lr = self.lr
            floating ld_reg = self.lambda_reg
            floating ld_bpr = self.lambda_bpr

        del_g1.fill(0)
        del_g2.fill(0)
        del_g3.fill(0)
        del_u.fill(0)
        del_i.fill(0)
        del_a.fill(0)
        del_o.fill(0)
        del_g1_reg.fill(0)
        del_g2_reg.fill(0)
        del_g3_reg.fill(0)
        del_u_reg.fill(0)
        del_i_reg.fill(0)
        del_a_reg.fill(0)
        del_o_reg.fill(0)
        with nogil, parallel(num_threads=n_threads):
            thread_id = get_thread_num()
            for _ in prange(num_samples, schedule='guided'):
                # get user item aspect element
                idx = rng_pos_uia.generate(thread_id)
                u_idx = X_uids[idx]
                i_idx = X_iids[idx]
                a_idx = X_aids[idx]
                score = X[idx]
                pred = get_score(G1, n_user_factors, n_item_factors, n_aspect_factors, U, I, A, u_idx, i_idx, a_idx)
                loss += (pred - score) * (pred - score)

                del_sqerror = 2 * (pred - score)
                for i in range(n_user_factors):
                    for j in range(n_item_factors):
                        for k in range(n_aspect_factors):
                            del_g1[i, j, k] += del_sqerror * U[u_idx, i] * I[i_idx, j] * A[a_idx, k]
                            del_u[u_idx, i] += del_sqerror * G1[i, j, k] * I[i_idx, j] * A[a_idx, k]
                            del_i[i_idx, j] += del_sqerror * G1[i, j, k] * U[u_idx, i] * A[a_idx, k]
                            del_a[a_idx, k] += del_sqerror * G1[i, j, k] * U[u_idx, i] * I[i_idx, j]

                # get user aspect opinion element
                idx = rng_pos_uao.generate(thread_id)
                u_idx = YU_uids[idx]
                a_idx = YU_aids[idx]
                o_idx = YU_oids[idx]
                score = YU[idx]
                pred = get_score(G2, n_user_factors, n_aspect_factors, n_opinion_factors, U, A, O, u_idx, a_idx, o_idx)
                loss += (pred - score) * (pred - score)

                del_sqerror = 2 * (pred - score)

                for i in range(n_user_factors):
                    for j in range(n_aspect_factors):
                        for k in range(n_opinion_factors):
                            del_g2[i, j, k] += del_sqerror * U[u_idx, i] * A[a_idx, j] * O[o_idx, k]
                            del_u[u_idx, i] += del_sqerror * G2[i, j, k] * A[a_idx, j] * O[o_idx, k]
                            del_a[a_idx, j] += del_sqerror * G2[i, j, k] * U[u_idx, i] * O[o_idx, k]
                            del_o[o_idx, k] += del_sqerror * G2[i, j, k] * U[u_idx, i] * A[a_idx, j]

                # get item aspect opinion element
                idx = rng_pos_iao.generate(thread_id)
                i_idx = YI_iids[idx]
                a_idx = YI_aids[idx]
                o_idx = YI_oids[idx]
                score = YI[idx]
                pred = get_score(G3, n_item_factors, n_aspect_factors, n_opinion_factors, I, A, O, i_idx, a_idx, o_idx)
                loss += (pred - score) * (pred - score)

                del_sqerror = 2 * (pred - score)
                for i in range(n_item_factors):
                    for j in range(n_aspect_factors):
                        for k in range(n_opinion_factors):
                            del_g3[i, j, k] += del_sqerror * I[i_idx, i] * A[a_idx, j] * O[o_idx, k]
                            del_i[i_idx, i] += del_sqerror * G3[i, j, k] * A[a_idx, j] * O[o_idx, k]
                            del_a[a_idx, j] += del_sqerror * G3[i, j, k] * I[i_idx, i] * O[o_idx, k]
                            del_o[o_idx, k] += del_sqerror * G3[i, j, k] * I[i_idx, i] * A[a_idx, j]

            for _ in prange(num_bpr_samples, schedule='guided'):
                idx = rng_pos.generate(thread_id)
                u_idx = user_ids[idx]
                i_idx = item_ids[idx]
                jdx = rng_neg.generate(thread_id)
                j_idx = neg_item_ids[jdx]

                s = 1
                # if the user has rated the item j, change sign if item j > item i
                if has_non_zero(indptr, item_ids, u_idx, j_idx):
                    i_score = rating_dict.my_map[get_key(u_idx, i_idx)]
                    j_score = rating_dict.my_map[get_key(u_idx, j_idx)]
                    if i_score == j_score:
                        skipped += 1
                        continue
                    elif i_score < j_score:
                        s = -1

                pred = (
                    get_score(G1, n_user_factors, n_item_factors, n_aspect_factors, U, I, A, u_idx, i_idx, n_aspects)
                    - get_score(G1, n_user_factors, n_item_factors, n_aspect_factors, U, I, A, u_idx, j_idx, n_aspects)
                ) * s
                z = (1.0 / (1.0 + exp(pred)))
                if z < .5:
                    correct += 1
                del_bpr = ld_bpr * z * s

                bpr_loss += log(1 / (1 + exp(-pred)))

                for i in range(n_user_factors):
                    for j in range(n_item_factors):
                        i_ij = I[i_idx, j] - I[j_idx, j]
                        for k in range(n_aspect_factors):
                            del_g1[i, j, k] -= del_bpr * U[u_idx, i] * i_ij * A[n_aspects, k]
                            del_u[u_idx, i] -= del_bpr * G1[i, j, k] * i_ij * A[n_aspects, k]
                            del_i[i_idx, j] -= del_bpr * G1[i, j, k] * U[u_idx, i] * A[n_aspects, k]
                            del_i[j_idx, j] += del_bpr * G1[i, j, k] * U[u_idx, i] * A[n_aspects, k]
                            del_a[n_aspects, k] -= del_bpr * G1[i, j, k] * U[u_idx, i] * i_ij

        # update params with AdaGrad
        with nogil, parallel(num_threads=n_threads):
            for i in prange(n_user_factors, schedule='guided'):
                for j in range(n_users):
                    if del_u[j, i] != 0:
                        del_u_reg[j, i] += del_u[j, i] + ld_reg * U[j, i]
                    sgrad_U[j, i] += eps + del_u_reg[j, i] * del_u_reg[j, i]
                    U[j, i] -= (lr / sqrt(sgrad_U[j, i])) * del_u_reg[j, i]
                    if U[j, i] < 0:
                        U[j, i] = 0

                for j in range(n_item_factors):
                    for k in range(n_aspect_factors):
                        if del_g1[i, j, k] != 0:
                            del_g1_reg[i, j, k] += del_g1[i, j, k] + ld_reg * G1[i, j, k]
                        sgrad_G1[i, j, k] += eps + del_g1_reg[i, j, k] * del_g1_reg[i, j, k]
                        G1[i, j, k] -= (lr / sqrt(sgrad_G1[i, j, k])) * del_g1_reg[i, j, k]
                        if G1[i, j, k] < 0:
                            G1[i, j, k] = 0

                for j in range(n_aspect_factors):
                    for k in range(n_opinion_factors):
                        if del_g2[i, j, k] != 0:
                            del_g2_reg[i, j, k] += del_g2[i, j, k] + ld_reg * G2[i, j, k]
                        sgrad_G2[i, j, k] += eps + del_g2_reg[i, j, k] * del_g2_reg[i, j, k]
                        G2[i, j, k] -= (lr / sqrt(sgrad_G2[i, j, k])) * del_g2_reg[i, j, k]
                        if G2[i, j, k] < 0:
                            G2[i, j, k] = 0

            for i in prange(n_item_factors, schedule='guided'):
                for j in range(n_items):
                    if del_i[j, i] != 0:
                        del_i_reg[j, i] += del_i[j, i] + ld_reg * I[j, i]
                    sgrad_I[j, i] += eps + del_i_reg[j, i] * del_i_reg[j, i]
                    I[j, i] -= (lr / sqrt(sgrad_I[j, i])) * del_i_reg[j, i]
                    if I[j, i] < 0:
                        I[j, i] = 0

                for j in range(n_aspect_factors):
                    for k in range(n_opinion_factors):
                        if del_g3[i, j, k] != 0:
                            del_g3_reg[i, j, k] += del_g3[i, j, k] + ld_reg * G3[i, j, k]
                        sgrad_G3[i, j, k] += eps + del_g3_reg[i, j, k] * del_g3_reg[i, j, k]
                        G3[i, j, k] -= (lr / sqrt(sgrad_G3[i, j, k])) * del_g3_reg[i, j, k]
                        if G3[i, j, k] < 0:
                            G3[i, j, k] = 0

            for i in prange(n_aspects + 1, schedule='guided'):
                for j in range(n_aspect_factors):
                    if del_a[i, j] != 0:
                        del_a_reg[i, j] += del_a[i, j] + ld_reg * A[i, j]
                    sgrad_A[i, j] += eps + del_a_reg[i, j] * del_a_reg[i, j]
                    A[i, j] -= (lr / sqrt(sgrad_A[i, j])) * del_a_reg[i, j]
                    if A[i, j] < 0:
                        A[i, j] = 0

            for i in prange(n_opinions, schedule='guided'):
                for j in range(n_opinion_factors):
                    if del_o[i, j] != 0:
                        del_o_reg[i, j] += del_o[i, j] + ld_reg * O[i, j]
                    sgrad_O[i, j] += eps + del_o_reg[i, j] * del_o_reg[i, j]
                    O[i, j] -= (lr / sqrt(sgrad_O[i, j])) * del_o_reg[i, j]
                    if O[i, j] < 0:
                        O[i, j] = 0

        return correct, skipped, loss, bpr_loss

    def score(self, u_idx, i_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        u_idx: int, required
            The index of the user for whom to perform score prediction.

        i_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if i_idx is None:
            if self.train_set.is_unk_user(u_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d" & u_idx
                )
            tensor_value1 = np.einsum(
                "abc,Ma->Mbc",
                self.G1,
                self.U[u_idx, :].reshape(1, self.n_user_factors),
            )
            tensor_value2 = np.einsum("Mbc,Nb->MNc", tensor_value1, self.I)
            item_scores = np.einsum("MNc,c->MN", tensor_value2, self.A[-1]).flatten()
            return item_scores
        else:
            if (self.train_set.is_unk_user(u_idx)
                or self.train_set.is_unk_item(i_idx)):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (u_idx, i_idx)
                )
            tensor_value1 = np.einsum("abc,a->bc", self.G1, self.U[u_idx])
            tensor_value2 = np.einsum("bc,b->c", tensor_value1, self.I[i_idx])
            item_score = np.einsum("c,c-> ", tensor_value2, self.A[-1])
            return item_score
