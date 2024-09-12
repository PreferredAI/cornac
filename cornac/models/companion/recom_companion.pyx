# Copyright 2024 The Cornac Authors. All Rights Reserved.
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
from cython.operator cimport dereference as deref
from cython.parallel import parallel, prange
from libc.math cimport sqrt, log, exp
from libcpp.map cimport map as cpp_map

import scipy.sparse as sp
import numpy as np
cimport numpy as np
from tqdm.auto import trange
from itertools import combinations
from collections import Counter, OrderedDict
from time import time

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.fast_dict cimport IntFloatDict
from ...utils.init_utils import uniform
from ..bpr.recom_bpr cimport RNGVector, has_non_zero
from ..mter.recom_mter cimport get_key, get_score

cdef extern from "../bpr/recom_bpr.h" namespace "recom_bpr" nogil:
    cdef int get_thread_num()

cdef int get_key3(int i_id, int j_id, int k_id) nogil:
    return get_key(get_key(i_id, j_id), k_id)


class Companion(Recommender):
    """Comparative Aspects and Opinions Ranking for Recommendation Explanations

    Parameters
    ----------
    name: string, optional, default: 'Companion'
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

    n_top_aspects: int, optional, default: 100
        The number of top scored aspects for each (user, item) pair to construct ranking score.

    alpha: float, optional, default: 0.5
        Trace off factor for constructing ranking score.

    lambda_reg: float, optional, default: 0.1
        The regularization parameter.

    lambda_bpr: float, optional, default: 10.0
        The regularization parameter for BPR.

    lambda_p: float, optional, default: 10.0
        The regularization parameter aspect ranking on item.

    lambda_a: float, optional, default: 10.0
        The regularization parameter for item ranking by aspect.

    lambda_y: float, optional, default: 10.0
        The regularization parameter for positive opinion ranking.

    lambda_z: float, optional, default: 10.0
        The regularization parameter for negative opinion ranking.

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
        List of initial parameters, e.g., init_params = {'U':U, 'I':I, 'A':A, 'O':O, 'O':O, 'G1':G1, 'G2':G2, 'G3':G3}

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    Trung-Hoang Le and Hady W. Lauw. 2024. \
    Learning to Rank Aspects and Opinions for Comparative Explanations. \
    Machine Learning (Special Issue for ACML 2024). \
    https://lthoang.com/assets/publications/mlj24.pdf
    """
    def __init__(
        self,
        name="Companion",
        rating_scale=5.0,
        n_user_factors=8,
        n_item_factors=8,
        n_aspect_factors=8,
        n_opinion_factors=8,
        n_bpr_samples=1000,
        n_aspect_ranking_samples=1000,
        n_opinion_ranking_samples=1000,
        n_element_samples=50,
        n_top_aspects=100,
        alpha=0.5,
        min_user_freq=2,
        min_pair_freq=1,
        min_common_freq=1,
        use_item_aspect_popularity=True,
        enum_window=None,
        lambda_reg=0.1,
        lambda_p=10,
        lambda_a=10,
        lambda_y=10,
        lambda_z=10,
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
        self.n_aspect_ranking_samples = n_aspect_ranking_samples
        self.n_opinion_ranking_samples = n_opinion_ranking_samples
        self.n_element_samples = n_element_samples
        self.n_top_aspects = n_top_aspects
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.lambda_bpr = lambda_bpr
        self.lambda_p = lambda_p
        self.lambda_a = lambda_a
        self.lambda_y = lambda_y
        self.lambda_z = lambda_z
        self.use_item_aspect_popularity = use_item_aspect_popularity
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
            G2_shape = (self.n_user_factors + self.n_item_factors, self.n_aspect_factors, self.n_opinion_factors)
            self.G2 = uniform(G2_shape, random_state=self.rng)
        if self.G3 is None:
            G3_shape = (self.n_user_factors + self.n_item_factors, self.n_aspect_factors, self.n_opinion_factors)
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

    def _build_item_quality_matrix(self, data_set, sentiment):
        start_time = time()
        quality_scores = []
        map_iid = []
        map_aspect_id = []
        for iid, sentiment_tup_ids_by_user in sentiment.item_sentiment.items():
            if self.is_unknown_item(iid):
                continue
            item_aspects = [tup[0]
                            for tup_id in sentiment_tup_ids_by_user.values()
                            for tup in sentiment.sentiment[tup_id]]
            item_aspect_count = Counter(item_aspects)
            total_sentiment_by_aspect = OrderedDict()
            for tup_id in sentiment_tup_ids_by_user.values():
                for aid, _, sentiment_polarity in sentiment.sentiment[tup_id]:
                    total_sentiment_by_aspect[aid] = total_sentiment_by_aspect.get(aid, 0) + sentiment_polarity
            for aid, total_sentiment in total_sentiment_by_aspect.items():
                map_iid.append(iid)
                map_aspect_id.append(aid)
                if self.use_item_aspect_popularity:
                    quality_scores.append(self._compute_quality_score(total_sentiment))
                else:
                    avg_sentiment = total_sentiment / item_aspect_count[aid]
                    quality_scores.append(self._compute_quality_score(avg_sentiment))
        quality_scores = np.asarray(quality_scores, dtype=np.float32).flatten()
        map_iid = np.asarray(map_iid, dtype=np.int32).flatten()
        map_aspect_id = np.asarray(map_aspect_id, dtype=np.int32).flatten()
        Y = sp.csr_matrix((quality_scores, (map_iid, map_aspect_id)),
                          shape=(self.train_set.num_items, sentiment.num_aspects))

        if self.verbose:
            print('Building item aspect quality matrix completed in %d s' % (time() - start_time))

        return Y

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
        user_item_aspect_pos_opinion = {}
        user_item_aspect_neg_opinion = {}

        for u_idx, sentiment_tup_ids_by_item in sentiment.user_sentiment.items():
            if self.is_unknown_user(u_idx):
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
                    user_item_aspect_pos_opinion.setdefault((u_idx, i_idx, a_idx, o_idx), 0)
                    user_item_aspect_neg_opinion.setdefault((u_idx, i_idx, a_idx, o_idx), 0)
                    if polarity > 0:
                        user_item_aspect_pos_opinion[(u_idx, i_idx, a_idx, o_idx)] += polarity
                    elif polarity < 0:
                        user_item_aspect_neg_opinion[(u_idx, i_idx, a_idx, o_idx)] += np.abs(polarity)

        user_item_aspect_keys = []
        user_item_aspect_scores = []
        for key in user_item_aspect.keys():
            uid, iid, aid = key
            if key[2] != sentiment.num_aspects:
                user_item_aspect[key] = self._compute_quality_score(
                    user_item_aspect[key]
                )
                user_item_aspect_keys.append(get_key3(uid, iid, aid))
                user_item_aspect_scores.append(user_item_aspect[key])
        user_item_aspect_dict = IntFloatDict(np.array(user_item_aspect_keys, dtype=np.intp), np.array(user_item_aspect_scores, dtype=np.float64))

        user_item_aspect_pos_opinion_keys = []
        user_item_aspect_pos_opinion_scores = []
        for key in user_item_aspect_pos_opinion.keys():
            uid, iid, aid, oid = key
            user_item_aspect_pos_opinion[key] = self._compute_attention_score(
                user_item_aspect_pos_opinion[key]
            )
            user_item_aspect_pos_opinion_keys.append(get_key3(get_key(uid, iid), aid, oid))
            user_item_aspect_pos_opinion_scores.append(user_item_aspect_pos_opinion[key])
        user_item_aspect_pos_opinion_dict = IntFloatDict(np.array(user_item_aspect_pos_opinion_keys, dtype=np.intp), np.array(user_item_aspect_pos_opinion_scores, dtype=np.float64))

        user_item_aspect_neg_opinion_keys = []
        user_item_aspect_neg_opinion_scores = []
        for key in user_item_aspect_neg_opinion.keys():
            uid, iid, aid, oid = key
            user_item_aspect_neg_opinion[key] = self._compute_attention_score(
                user_item_aspect_neg_opinion[key]
            )
            user_item_aspect_neg_opinion_keys.append(get_key3(get_key(uid, iid), aid, oid))
            user_item_aspect_neg_opinion_scores.append(user_item_aspect_neg_opinion[key])
        user_item_aspect_neg_opinion_dict = IntFloatDict(np.array(user_item_aspect_neg_opinion_keys, dtype=np.intp), np.array(user_item_aspect_neg_opinion_scores, dtype=np.float64))

        Y = self._build_item_quality_matrix(data_set, sentiment)

        if self.verbose:
            total_time = time.time() - start_time
            print("Building data completed in %d s" % total_time)
        return (
            rating_matrix,
            rating_dict,
            user_item_aspect,
            user_item_aspect_pos_opinion,
            user_item_aspect_neg_opinion,
            user_item_aspect_dict,
            user_item_aspect_pos_opinion_dict,
            user_item_aspect_neg_opinion_dict,
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
            user_item_aspect_pos_opinion,
            user_item_aspect_neg_opinion,
            user_item_aspect_dict,
            user_item_aspect_pos_opinion_dict,
            user_item_aspect_neg_opinion_dict,
        ) = self._build_data(train_set)
        self.user_item_aspect_dict = user_item_aspect_dict
        self.user_item_aspect_pos_opinion_dict = user_item_aspect_pos_opinion_dict
        self.user_item_aspect_neg_opinion_dict = user_item_aspect_neg_opinion_dict
        X, X_uids, X_iids, X_aids = [], [], [], []
        YP, YP_uids, YP_iids, YP_aids, YP_oids = [], [], [], [], []
        YN, YN_uids, YN_iids, YN_aids, YN_oids = [], [], [], [], []

        for (uid, iid, aid), score in user_item_aspect.items():
            X.append(score)
            X_uids.append(uid)
            X_iids.append(iid)
            X_aids.append(aid)

        for (uid, iid, aid, oid), score in user_item_aspect_pos_opinion.items():
            YP.append(score)
            YP_uids.append(uid)
            YP_iids.append(iid)
            YP_aids.append(aid)
            YP_oids.append(oid)

        for (uid, iid, aid, oid), score in user_item_aspect_neg_opinion.items():
            YN.append(score)
            YN_uids.append(uid)
            YN_iids.append(iid)
            YN_aids.append(aid)
            YN_oids.append(oid)

        X = np.array(X, dtype=np.float32)
        X_uids = np.array(X_uids, dtype=np.int32)
        X_iids = np.array(X_iids, dtype=np.int32)
        X_aids = np.array(X_aids, dtype=np.int32)
        YP = np.array(YP, dtype=np.float32)
        YP_uids = np.array(YP_uids, dtype=np.int32)
        YP_iids = np.array(YP_iids, dtype=np.int32)
        YP_aids = np.array(YP_aids, dtype=np.int32)
        YP_oids = np.array(YP_oids, dtype=np.int32)
        YN = np.array(YN, dtype=np.float32)
        YN_uids = np.array(YN_uids, dtype=np.int32)
        YN_iids = np.array(YN_iids, dtype=np.int32)
        YN_aids = np.array(YN_aids, dtype=np.int32)
        YN_oids = np.array(YN_oids, dtype=np.int32)

        user_counts = np.ediff1d(rating_matrix.indptr).astype(np.int32)
        user_ids = np.repeat(np.arange(self.train_set.num_users), user_counts).astype(np.int32)
        neg_item_ids = np.arange(train_set.num_items, dtype=np.int32)

        cdef:
            int n_threads = self.n_threads
            RNGVector rng_pos_uia = RNGVector(n_threads, len(user_item_aspect) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_pos_uiaop = RNGVector(n_threads, len(user_item_aspect_pos_opinion) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_pos_uiaon = RNGVector(n_threads, len(user_item_aspect_neg_opinion) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_pos = RNGVector(n_threads, len(user_ids) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_neg = RNGVector(n_threads, train_set.num_items - 1, self.rng.randint(2 ** 31))
            RNGVector rng_neg_uia = RNGVector(n_threads, self.train_set.sentiment.num_aspects - 1, self.rng.randint(2 ** 31))
            RNGVector rng_neg_uiaop = RNGVector(n_threads, self.train_set.sentiment.num_opinions - 1, self.rng.randint(2 ** 31))
            RNGVector rng_neg_uiaon = RNGVector(n_threads, self.train_set.sentiment.num_opinions - 1, self.rng.randint(2 ** 31))

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
                correct, skipped, loss, bpr_loss = self._fit(
                    rng_pos, rng_neg,
                    rng_pos_uia, rng_neg_uia,
                    rng_pos_uiaop, rng_pos_uiaon,
                    rng_neg_uiaop, rng_neg_uiaon,
                    n_threads,
                    X, X_uids, X_iids, X_aids,
                    YP, YP_uids, YP_iids, YP_aids, YP_oids,
                    YN, YN_uids, YN_iids, YN_aids, YN_oids,
                    rating_dict,
                    user_item_aspect_dict,
                    user_item_aspect_pos_opinion_dict,
                    user_item_aspect_neg_opinion_dict,
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
                    "skipped": "%.2f%%" % (100.0 * skipped / self.n_bpr_samples),
                })

        if self.verbose:
            print('Optimization finished!')

        return self

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit(
        self,
        RNGVector rng_pos,
        RNGVector rng_neg,
        RNGVector rng_pos_uia,
        RNGVector rng_neg_uia,
        RNGVector rng_pos_uiaop,
        RNGVector rng_pos_uiaon,
        RNGVector rng_neg_uiaop,
        RNGVector rng_neg_uiaon,
        int n_threads,
        floating[:] X, integral[:] X_uids, integral[:] X_iids, integral[:] X_aids,
        floating[:] YP, integral[:] YP_uids, integral[:] YP_iids, integral[:] YP_aids, integral[:] YP_oids,
        floating[:] YN, integral[:] YN_uids, integral[:] YN_iids, integral[:] YN_aids, integral[:] YN_oids,
        IntFloatDict rating_dict,
        IntFloatDict user_item_aspect_dict,
        IntFloatDict user_item_aspect_pos_opinion_dict,
        IntFloatDict user_item_aspect_neg_opinion_dict,
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
        np.ndarray[np.float32_t, ndim=2] del_o_reg,
        ):
        """Fit the model parameters (G1, G2, G3, U, I, A, O)
        """
        cdef:
            long s, i_index, j_index
            long correct = 0, correct_uia = 0, correct_uia_i = 0, correct_uiaop = 0, correct_uiaon = 0, aspect_correct = 0
            long skipped = 0, skipped_uia = 0, skipped_uia_i = 0, skipped_uiaop = 0, skipped_uiaon = 0
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
            int num_aspect_ranking_samples = self.n_aspect_ranking_samples
            int num_opinion_ranking_samples = self.n_opinion_ranking_samples

            integral _, i, j, k, idx, jdx, u_idx, i_idx, i_jdx, a_idx, a_jdx, o_idx, o_jdx, j_idx, thread_id
            floating z, score, i_score, j_score, pred, temp, i_ij, uia_ij, uia_ij_i, uiaop_ij, uiaon_ij, a_ji
            floating loss = 0., bpr_loss = 0., bpr_loss_uia = 0., bpr_loss_uia_i = 0., bpr_loss_uiaop = 0., bpr_loss_uiaon = 0., aspect_bpr_loss = 0.
            floating del_sqerror, del_s_uia, del_bpr, del_bpr_uia, del_bpr_uia_i, del_bpr_uiaop, del_bpr_uiaon, del_aspect_bpr
            floating eps = 1e-9

            floating lr = self.lr
            floating ld_reg = self.lambda_reg
            floating ld_bpr = self.lambda_bpr
            floating ld_p = self.lambda_p
            floating ld_a = self.lambda_a
            floating ld_y = self.lambda_y
            floating ld_z = self.lambda_z
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

                # get user aspect positive opinion element
                idx = rng_pos_uiaop.generate(thread_id)
                u_idx = YP_uids[idx]
                i_idx = YP_iids[idx]
                a_idx = YP_aids[idx]
                o_idx = YP_oids[idx]
                score = YP[idx]
                pred = (
                    get_score(G2, n_user_factors, n_aspect_factors, n_opinion_factors, U, A, O, u_idx, a_idx, o_idx)
                    + get_score(G2[n_user_factors:, :, :], n_item_factors, n_aspect_factors, n_opinion_factors, I, A, O, i_idx, a_idx, o_idx)
                )
                loss += (pred - score) * (pred - score)

                del_sqerror = 2 * (pred - score)

                for i in range(n_user_factors):
                    for j in range(n_aspect_factors):
                        for k in range(n_opinion_factors):
                            del_g2[i, j, k] += del_sqerror * U[u_idx, i] * A[a_idx, j] * O[o_idx, k]
                            del_u[u_idx, i] += del_sqerror * G2[i, j, k] * A[a_idx, j] * O[o_idx, k]
                            del_a[a_idx, j] += del_sqerror * G2[i, j, k] * U[u_idx, i] * O[o_idx, k]
                            del_o[o_idx, k] += del_sqerror * G2[i, j, k] * U[u_idx, i] * A[a_idx, j]

                for i in range(n_item_factors):
                    for j in range(n_aspect_factors):
                        for k in range(n_opinion_factors):
                            del_g2[i, j, k] += del_sqerror * I[i_idx, i] * A[a_idx, j] * O[o_idx, k]
                            del_i[i_idx, i] += del_sqerror * G2[i, j, k] * A[a_idx, j] * O[o_idx, k]
                            del_a[a_idx, j] += del_sqerror * G2[i, j, k] * I[i_idx, i] * O[o_idx, k]
                            del_o[o_idx, k] += del_sqerror * G2[i, j, k] * I[i_idx, i] * A[a_idx, j]

                # get user aspect negative opinion element
                idx = rng_pos_uiaon.generate(thread_id)
                u_idx = YN_uids[idx]
                i_idx = YN_iids[idx]
                a_idx = YN_aids[idx]
                o_idx = YN_oids[idx]
                score = YN[idx]
                pred = (
                    get_score(G3, n_user_factors, n_aspect_factors, n_opinion_factors, U, A, O, u_idx, a_idx, o_idx)
                    + get_score(G3[n_user_factors:, :, :], n_item_factors, n_aspect_factors, n_opinion_factors, I, A, O, i_idx, a_idx, o_idx)
                )
                loss += (pred - score) * (pred - score)

                del_sqerror = 2 * (pred - score)

                for i in range(n_user_factors):
                    for j in range(n_aspect_factors):
                        for k in range(n_opinion_factors):
                            del_g3[i, j, k] += del_sqerror * U[u_idx, i] * A[a_idx, j] * O[o_idx, k]
                            del_u[u_idx, i] += del_sqerror * G3[i, j, k] * A[a_idx, j] * O[o_idx, k]
                            del_a[a_idx, j] += del_sqerror * G3[i, j, k] * U[u_idx, i] * O[o_idx, k]
                            del_o[o_idx, k] += del_sqerror * G3[i, j, k] * U[u_idx, i] * A[a_idx, j]

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


            for _ in prange(num_aspect_ranking_samples, schedule='guided'):
                idx = rng_pos_uia.generate(thread_id)
                u_idx = X_uids[idx]
                i_idx = X_iids[idx]
                a_idx = X_aids[idx]
                a_jdx = rng_neg_uia.generate(thread_id)
                s = 1
                if user_item_aspect_dict.my_map.find(get_key3(u_idx, i_idx, a_jdx)) != user_item_aspect_dict.my_map.end():
                    i_score = X[idx]
                    j_score = user_item_aspect_dict.my_map[get_key3(u_idx, i_idx, a_jdx)]
                    if i_score == j_score:
                        skipped_uia += 1
                        continue
                    elif i_score < j_score:
                        s = -1

                pred = (
                    get_score(G1, n_user_factors, n_item_factors, n_aspect_factors, U, I, A, u_idx, i_idx, a_idx)
                    - get_score(G1, n_user_factors, n_item_factors, n_aspect_factors, U, I, A, u_idx, i_idx, a_jdx)
                ) * s
                z = (1.0 / (1.0 + exp(pred)))
                if z < .5:
                   correct_uia += 1
                del_bpr_uia = ld_p * z * s

                bpr_loss_uia += log(1 / (1 + exp(-pred)))
                for k in range(n_aspect_factors):
                    uia_ij = A[a_idx, k] - A[a_jdx, k]
                    for i in range(n_user_factors):
                        for j in range(n_item_factors):
                            del_g1[i, j, k] -= del_bpr_uia * U[u_idx, i] * I[i_idx, j] * uia_ij
                            del_u[u_idx, i] -= del_bpr_uia * G1[i, j, k] * I[i_idx, j] * uia_ij
                            del_i[i_idx, j] -= del_bpr_uia * G1[i, j, k] * U[u_idx, i] * uia_ij
                            del_a[a_idx, k] -= del_bpr_uia * G1[i, j, k] * U[u_idx, i] * I[i_idx, j]
                            del_a[a_jdx, k] += del_bpr_uia * G1[i, j, k] * U[u_idx, i] * I[i_idx, j]


            for _ in prange(num_aspect_ranking_samples, schedule='guided'):
                idx = rng_pos_uia.generate(thread_id)
                u_idx = X_uids[idx]
                i_idx = X_iids[idx]
                a_idx = X_aids[idx]
                i_jdx = rng_neg.generate(thread_id)
                s = 1
                if user_item_aspect_dict.my_map.find(get_key3(u_idx, i_jdx, a_idx)) != user_item_aspect_dict.my_map.end():
                    i_score = X[idx]
                    j_score = user_item_aspect_dict.my_map[get_key3(u_idx, i_jdx, a_idx)]
                    if i_score == j_score:
                        skipped_uia_i += 1
                        continue
                    elif i_score < j_score:
                        s = -1

                pred = (
                    get_score(G1, n_user_factors, n_item_factors, n_aspect_factors, U, I, A, u_idx, i_idx, a_idx)
                    - get_score(G1, n_user_factors, n_item_factors, n_aspect_factors, U, I, A, u_idx, i_jdx, a_idx)
                ) * s
                z = (1.0 / (1.0 + exp(pred)))
                if z < .5:
                   correct_uia_i += 1
                del_bpr_uia_i = ld_a * z * s

                bpr_loss_uia_i += log(1 / (1 + exp(-pred)))
                for j in range(n_item_factors):
                    uia_ij_i = I[i_idx, j] - I[i_jdx, j]
                    for k in range(n_aspect_factors):
                        for i in range(n_user_factors):
                            del_g1[i, j, k] -= del_bpr_uia_i * U[u_idx, i] * uia_ij_i * A[a_idx, k]
                            del_u[u_idx, i] -= del_bpr_uia_i * G1[i, j, k] * uia_ij_i * A[a_idx, k]
                            del_i[i_idx, j] -= del_bpr_uia_i * G1[i, j, k] * U[u_idx, i] * A[a_idx, k]
                            del_i[i_jdx, j] += del_bpr_uia_i * G1[i, j, k] * U[u_idx, i] * A[a_idx, k]
                            del_a[a_idx, k] -= del_bpr_uia_i * G1[i, j, k] * U[u_idx, i] * uia_ij_i

            for _ in prange(num_opinion_ranking_samples, schedule='guided'):
                idx = rng_pos_uiaop.generate(thread_id)
                u_idx = YP_uids[idx]
                i_idx = YP_iids[idx]
                a_idx = YP_aids[idx]
                o_idx = YP_oids[idx]
                o_jdx = rng_neg_uiaop.generate(thread_id)
                s = 1
                if user_item_aspect_pos_opinion_dict.my_map.find(get_key3(get_key(u_idx, i_idx), a_idx, o_jdx)) != user_item_aspect_pos_opinion_dict.my_map.end():
                    i_score = YP[idx]
                    j_score = user_item_aspect_pos_opinion_dict.my_map[get_key3(get_key(u_idx, i_idx), a_idx, o_jdx)]
                    if i_score == j_score:
                        skipped_uiaop += 1
                        continue
                    elif i_score < j_score:
                        s = -1

                pred = (
                    get_score(G2, n_user_factors, n_aspect_factors, n_opinion_factors, U, A, O, u_idx, a_idx, o_idx)
                    + get_score(G2[n_user_factors:, :, :], n_item_factors, n_aspect_factors, n_opinion_factors, I, A, O, i_idx, a_idx, o_idx)
                    - get_score(G2, n_user_factors, n_aspect_factors, n_opinion_factors, U, A, O, u_idx, a_idx, o_jdx)
                    - get_score(G2[n_user_factors:, :, :], n_item_factors, n_aspect_factors, n_opinion_factors, I, A, O, i_idx, a_idx, o_jdx)
                ) * s
                z = (1.0 / (1.0 + exp(pred)))
                if z < .5:
                   correct_uiaop += 1
                del_bpr_uiaop = ld_y * z * s

                bpr_loss_uiaop += log(1 / (1 + exp(-pred)))
                for k in range(n_opinion_factors):
                    uiaop_ij = O[o_idx, k] - O[o_jdx, k]
                    for j in range(n_aspect_factors):
                        for i in range(n_user_factors):
                            del_g2[i, j, k] -= del_bpr_uiaop * U[u_idx, i] * A[a_idx, j] * uiaop_ij
                            del_u[u_idx, i] -= del_bpr_uiaop * G2[i, j, k] * A[a_idx, j] * uiaop_ij
                            del_a[a_idx, j] -= del_bpr_uiaop * G2[i, j, k] * U[u_idx, i] * uiaop_ij
                            del_o[o_idx, k] -= del_bpr_uiaop * G2[i, j, k] * U[u_idx, i] * A[a_idx, j]
                            del_o[o_jdx, k] += del_bpr_uiaop * G2[i, j, k] * U[u_idx, i] * A[a_idx, j]
                        for i in range(n_item_factors):
                            del_g2[n_user_factors + i, j, k] -= del_bpr_uiaop * I[i_idx, i] * A[a_idx, j] * uiaop_ij
                            del_i[i_idx, i] -= del_bpr_uiaop * G2[n_user_factors + i, j, k] * A[a_idx, j] * uiaop_ij
                            del_a[a_idx, j] -= del_bpr_uiaop * G2[n_user_factors + i, j, k] * I[i_idx, i] * uiaop_ij
                            del_o[o_idx, k] -= del_bpr_uiaop * G2[n_user_factors + i, j, k] * I[i_idx, i] * A[a_idx, j]
                            del_o[o_jdx, k] += del_bpr_uiaop * G2[n_user_factors + i, j, k] * I[i_idx, i] * A[a_idx, j]

            for _ in prange(num_opinion_ranking_samples, schedule='guided'):
                idx = rng_pos_uiaon.generate(thread_id)
                u_idx = YN_uids[idx]
                i_idx = YN_iids[idx]
                a_idx = YN_aids[idx]
                o_idx = YN_oids[idx]
                o_jdx = rng_neg_uiaon.generate(thread_id)
                s = 1
                if user_item_aspect_neg_opinion_dict.my_map.find(get_key3(get_key(u_idx, i_idx), a_idx, o_jdx)) != user_item_aspect_neg_opinion_dict.my_map.end():
                    i_score = YN[idx]
                    j_score = user_item_aspect_pos_opinion_dict.my_map[get_key3(get_key(u_idx, i_idx), a_idx, o_jdx)]
                    if i_score == j_score:
                        skipped_uiaon += 1
                        continue
                    elif i_score < j_score:
                        s = -1

                pred = (
                    get_score(G3, n_user_factors, n_aspect_factors, n_opinion_factors, U, A, O, u_idx, a_idx, o_idx)
                    + get_score(G3[n_user_factors:, :, :], n_item_factors, n_aspect_factors, n_opinion_factors, I, A, O, i_idx, a_idx, o_idx)
                    - get_score(G3, n_user_factors, n_aspect_factors, n_opinion_factors, U, A, O, u_idx, a_idx, o_jdx)
                    - get_score(G3[n_user_factors:, :, :], n_item_factors, n_aspect_factors, n_opinion_factors, I, A, O, i_idx, a_idx, o_jdx)
                ) * s
                z = (1.0 / (1.0 + exp(pred)))
                if z < .5:
                   correct_uiaon += 1
                del_bpr_uiaon = ld_z * z * s

                bpr_loss_uiaon += log(1 / (1 + exp(-pred)))
                for k in range(n_opinion_factors):
                    uiaon_ij = O[o_idx, k] - O[o_jdx, k]
                    for j in range(n_aspect_factors):
                        for i in range(n_user_factors):
                            del_g3[i, j, k] -= del_bpr_uiaon * U[u_idx, i] * A[a_idx, j] * uiaon_ij
                            del_u[u_idx, i] -= del_bpr_uiaon * G2[i, j, k] * A[a_idx, j] * uiaon_ij
                            del_a[a_idx, j] -= del_bpr_uiaon * G2[i, j, k] * U[u_idx, i] * uiaon_ij
                            del_o[o_idx, k] -= del_bpr_uiaon * G2[i, j, k] * U[u_idx, i] * A[a_idx, j]
                            del_o[o_jdx, k] += del_bpr_uiaon * G2[i, j, k] * U[u_idx, i] * A[a_idx, j]
                        for i in range(n_item_factors):
                            del_g3[n_user_factors + i, j, k] -= del_bpr_uiaon * I[i_idx, i] * A[a_idx, j] * uiaon_ij
                            del_i[i_idx, i] -= del_bpr_uiaon * G3[n_user_factors + i, j, k] * A[a_idx, j] * uiaon_ij
                            del_a[a_idx, j] -= del_bpr_uiaon * G3[n_user_factors + i, j, k] * I[i_idx, i] * uiaon_ij
                            del_o[o_idx, k] -= del_bpr_uiaon * G3[n_user_factors + i, j, k] * I[i_idx, i] * A[a_idx, j]
                            del_o[o_jdx, k] += del_bpr_uiaon * G3[n_user_factors + i, j, k] * I[i_idx, i] * A[a_idx, j]

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
                        if del_g3[i, j, k] != 0:
                            del_g3_reg[i, j, k] += del_g3[i, j, k] + ld_reg * G3[i, j, k]
                        sgrad_G3[i, j, k] += eps + del_g3_reg[i, j, k] * del_g3_reg[i, j, k]
                        G3[i, j, k] -= (lr / sqrt(sgrad_G3[i, j, k])) * del_g3_reg[i, j, k]
                        if G3[i, j, k] < 0:
                            G3[i, j, k] = 0

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
                        if del_g2[n_user_factors + i, j, k] != 0:
                            del_g2_reg[n_user_factors + i, j, k] += del_g2[n_user_factors + i, j, k] + ld_reg * G2[n_user_factors + i, j, k]
                        sgrad_G2[n_user_factors + i, j, k] += eps + del_g2_reg[n_user_factors + i, j, k] * del_g2_reg[n_user_factors + i, j, k]
                        G2[n_user_factors + i, j, k] -= (lr / sqrt(sgrad_G2[n_user_factors + i, j, k])) * del_g2_reg[n_user_factors + i, j, k]
                        if G2[n_user_factors + i, j, k] < 0:
                            G2[n_user_factors + i, j, k] = 0
                        if del_g3[n_user_factors + i, j, k] != 0:
                            del_g3_reg[n_user_factors + i, j, k] += del_g3[n_user_factors + i, j, k] + ld_reg * G3[n_user_factors + i, j, k]
                        sgrad_G3[n_user_factors + i, j, k] += eps + del_g3_reg[n_user_factors + i, j, k] * del_g3_reg[n_user_factors + i, j, k]
                        G3[n_user_factors + i, j, k] -= (lr / sqrt(sgrad_G3[n_user_factors + i, j, k])) * del_g3_reg[n_user_factors + i, j, k]
                        if G3[n_user_factors + i, j, k] < 0:
                            G3[n_user_factors + i, j, k] = 0

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
            if self.is_unknown_user(u_idx):
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
            if (self.is_unknown_user(u_idx)
                or self.is_unknown_item(i_idx)):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (u_idx, i_idx)
                )
            tensor_value1 = np.einsum("abc,a->bc", self.G1, self.U[u_idx])
            tensor_value2 = np.einsum("bc,b->c", tensor_value1, self.I[i_idx])
            item_score = np.einsum("c,c-> ", tensor_value2, self.A[-1])
            return item_score


    def rank(self, user_idx, item_indices=None, k=-1):
        if self.alpha > 0 and self.n_top_aspects > 0:
            n_items = self.train_set.num_items
            n_top_aspects = min(self.n_top_aspects, self.train_set.sentiment.num_aspects)
            item_indices = item_indices if item_indices is not None else range(n_items)
            item_indices = np.array(item_indices)
            ts1 = np.einsum("abc,a->bc", self.G1, self.U[user_idx])
            ts2 = np.einsum("bc,Mb->Mc", ts1, self.I[item_indices])
            ts3 = np.einsum("Mc,Nc->MN", ts2, self.A)
            top_aspect_scores = ts3[
                np.repeat(range(n_items), n_top_aspects).reshape(
                    n_items, n_top_aspects
                ),
                ts3[:, :-1].argsort(axis=1)[::-1][:, :n_top_aspects],
            ]
            item_scores = (
                self.alpha * top_aspect_scores.mean(axis=1) + (1 - self.alpha) * ts3[:, -1]
            )

            if k != -1:  # O(n + k log k), faster for small k which is usually the case
                partitioned_idx = np.argpartition(item_scores, -k)
                top_k_idx = partitioned_idx[-k:]
                sorted_top_k_idx = top_k_idx[np.argsort(item_scores[top_k_idx])]
                partitioned_idx[-k:] = sorted_top_k_idx
                ranked_items = item_indices[partitioned_idx[::-1]]
            else:  # O(n log n)
                ranked_items = item_indices[item_scores.argsort()[::-1]]
            return ranked_items, item_scores
        return super().rank(user_idx, item_indices, k)