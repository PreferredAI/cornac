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
from tqdm import trange
from itertools import combinations
from collections import Counter, OrderedDict
from time import time

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.fast_dict cimport IntFloatDict
from ...utils.init_utils import uniform
from ..bpr.recom_bpr cimport RNGVector, has_non_zero
from ..mter import MTER
from ..mter.recom_mter cimport get_key, get_score


cdef extern from "../bpr/recom_bpr.h" namespace "recom_bpr" nogil:
    cdef int get_thread_num()


class ComparERSub(MTER):
    """Explainable Recommendation with Comparative Constraints on Subjective Aspect-Level Quality

    Parameters
    ----------
    name: string, optional, default: 'ComparERSub'
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
    * Trung-Hoang Le and Hady W. Lauw. "Explainable Recommendation with Comparative Constraints on Product Aspects."
    ACM International Conference on Web Search and Data Mining (WSDM). 2021.
    """
    def __init__(
        self,
        name="ComparERSub",
        rating_scale=5.0,
        n_user_factors=8,
        n_item_factors=8,
        n_aspect_factors=8,
        n_opinion_factors=8,
        n_pair_samples=1000,
        n_bpr_samples=1000,
        n_element_samples=50,
        min_user_freq=2,
        min_pair_freq=1,
        min_common_freq=1,
        use_item_aspect_popularity=True,
        enum_window=None,
        lambda_reg=0.1,
        lambda_bpr=10,
        lambda_d=0.01,
        max_iter=200000,
        lr=0.5,
        n_threads=0,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, rating_scale = rating_scale, n_user_factors = n_user_factors, n_item_factors = n_item_factors, n_aspect_factors = n_aspect_factors, n_opinion_factors = n_opinion_factors, n_bpr_samples = n_bpr_samples, n_element_samples = n_element_samples, lambda_reg = lambda_reg, lambda_bpr = lambda_bpr, max_iter = max_iter, lr = lr, n_threads=n_threads, seed = seed, trainable=trainable, init_params=init_params, verbose=verbose)
        self.lambda_d = lambda_d
        self.n_pair_samples = n_pair_samples
        self.min_user_freq = min_user_freq
        self.min_pair_freq = min_pair_freq
        self.min_common_freq = min_common_freq
        self.use_item_aspect_popularity = use_item_aspect_popularity
        self.enum_window = enum_window


    def _build_item_quality_matrix(self, data_set, sentiment):
        start_time = time()
        quality_scores = []
        map_iid = []
        map_aspect_id = []
        for iid, sentiment_tup_ids_by_user in sentiment.item_sentiment.items():
            if self.train_set.is_unk_item(iid):
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
        from tqdm import tqdm

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
        for u_idx, sentiment_tup_ids_by_item in tqdm(sentiment.user_sentiment.items(), disable=not self.verbose, desc='Count aspects'):
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

        for key in tqdm(user_item_aspect.keys(), disable=not self.verbose, desc='Compute X'):
            if key[2] != sentiment.num_aspects:
                user_item_aspect[key] = self._compute_quality_score(
                    user_item_aspect[key]
                )

        for key in tqdm(user_aspect_opinion.keys(), disable=not self.verbose, desc='Compute YU'):
            user_aspect_opinion[key] = self._compute_attention_score(
                user_aspect_opinion[key]
            )

        for key in tqdm(item_aspect_opinion.keys(), disable=not self.verbose, desc='Compute YI'):
            item_aspect_opinion[key] = self._compute_attention_score(
                item_aspect_opinion[key]
            )
        Y = self._build_item_quality_matrix(data_set, sentiment)
        user_indices, earlier_indices, later_indices, aspect_indices, pair_freq = self._build_chrono_purchased_pairs(data_set, user_item_aspect, Y)

        if self.verbose:
            total_time = time.time() - start_time
            print("Building data completed in %d s" % total_time)
        return (
            rating_matrix,
            rating_dict,
            user_item_aspect,
            user_aspect_opinion,
            item_aspect_opinion,
            user_indices, earlier_indices, later_indices, aspect_indices, pair_freq
        )

    def _build_chrono_purchased_pairs(self, data_set, user_item_aspect, Y):
        start_time = time()
        from tqdm import tqdm
        chrono_purchased_pairs = Counter()
        for user_idx, (item_ids, *_) in tqdm(data_set.chrono_user_data.items(), disable=not self.verbose, desc='Get purchased pairs'):
            if len(item_ids) >= self.min_user_freq:
                window = len(item_ids) if self.enum_window is None else min(self.enum_window, len(item_ids))
                for sub_item_ids in [item_ids[i:i+window] for i in range(len(item_ids) - window + 1)]:
                    for earlier_item_idx, later_item_idx in combinations(sub_item_ids, 2):
                        if self.train_set.is_unk_item(earlier_item_idx) or self.train_set.is_unk_item(later_item_idx):
                            continue
                        chrono_purchased_pairs[(user_idx, earlier_item_idx, later_item_idx)] += 1

        pair_counts = Counter()
        common_aspect_counts = Counter()
        counted_pairs = set()
        not_dominated_pairs = set()
        for (user_idx, earlier_item_idx, later_item_idx), count in tqdm(chrono_purchased_pairs.most_common(), disable=not self.verbose, desc='Get skyline aspects'):
            for k in range(self.train_set.sentiment.num_aspects - 1): # ignore rating at the last index
                if user_item_aspect.get((user_idx, later_item_idx, k), 0) > user_item_aspect.get((user_idx, earlier_item_idx, k), 0):
                    pair_counts[(user_idx, earlier_item_idx, later_item_idx, k)] += count
                    not_dominated_pairs.add((user_idx, earlier_item_idx, later_item_idx))
                if Y[earlier_item_idx, k] > 0 and Y[later_item_idx, k] > 0 and (earlier_item_idx, later_item_idx) not in counted_pairs:
                    common_aspect_counts[(earlier_item_idx, later_item_idx)] += 1
            if (earlier_item_idx, later_item_idx) not in counted_pairs:
                counted_pairs.add((earlier_item_idx, later_item_idx))

        user_indices, earlier_indices, later_indices, aspect_indices, pair_freq = [], [], [], [], []
        for (user_idx, earlier_item_idx, later_item_idx, aspect_idx), count in tqdm(pair_counts.most_common(), disable=not self.verbose, desc='Enumerate index'):
            if common_aspect_counts[(earlier_item_idx, later_item_idx)] < self.min_common_freq:
                continue
            user_indices.append(user_idx)
            earlier_indices.append(earlier_item_idx)
            later_indices.append(later_item_idx)
            aspect_indices.append(aspect_idx)
            pair_freq.append(count)

        user_indices = np.asarray(user_indices, dtype=np.int32).flatten()
        earlier_indices = np.asarray(earlier_indices, dtype=np.int32).flatten()
        later_indices = np.asarray(later_indices, dtype=np.int32).flatten()
        aspect_indices = np.asarray(aspect_indices, dtype=np.int32).flatten()
        pair_freq = np.asarray(pair_freq, dtype=np.int32).flatten()

        if self.verbose:
            print('Building chrono purchared items pairs completed in %d s' % (time() - start_time))
            print('Statistics: # aspect pairs >= %d = %d, min(%.2f), max(%.2f), avg(%.2f)' % (
                self.min_pair_freq,
                pair_freq[pair_freq>=self.min_pair_freq].shape[0],
                pair_freq.min(),
                pair_freq.max(),
                pair_freq.mean()))
            print('# earlier-later pairs: %d, # unique earlier-later pairs: %d, not dominated pairs %d, # comparable pairs %d' % (
                sum(chrono_purchased_pairs.values()),
                len(chrono_purchased_pairs),
                len(not_dominated_pairs),
                len(pair_freq)
            ))

        return user_indices, earlier_indices, later_indices, aspect_indices, pair_freq



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
            p_user_indices, earlier_indices, later_indices, aspect_indices, pair_freq
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
            RNGVector rng_pos_pair = RNGVector(n_threads, len(p_user_indices) - 1, self.rng.randint(2 ** 31))
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
                    rng_pos_uia, rng_pos_uao, rng_pos_iao, rng_pos_pair, rng_pos, rng_neg,
                    n_threads,
                    X, X_uids, X_iids, X_aids,
                    YU, YU_uids, YU_aids, YU_oids,
                    YI, YI_iids, YI_aids, YI_oids,
                    rating_dict,
                    user_ids, rating_matrix.indices, neg_item_ids, rating_matrix.indptr,
                    p_user_indices, earlier_indices, later_indices, aspect_indices, pair_freq,
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
        RNGVector rng_pos_pair,
        RNGVector rng_pos,
        RNGVector rng_neg,
        int n_threads,
        floating[:] X, integral[:] X_uids, integral[:] X_iids, integral[:] X_aids,
        floating[:] YU, integral[:] YU_uids, integral[:] YU_aids, integral[:] YU_oids,
        floating[:] YI, integral[:] YI_iids, integral[:] YI_aids, integral[:] YI_oids,
        IntFloatDict rating_dict,
        integral[:] user_ids, integral[:] item_ids, integral[:] neg_item_ids, integral[:] indptr,
        integral[:] p_user_indices, integral[:] earlier_indices, integral[:] later_indices, integral[:] aspect_indices, integral[:] pair_counts,
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
            long s, i_index, j_index, correct = 0, skipped = 0, aspect_correct = 0
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
            int num_pair_samples = self.n_pair_samples

            integral _, i, j, k, idx, jdx, u_idx, i_idx, a_idx, o_idx, j_idx, thread_id
            floating z, score, i_score, j_score, pred, temp, i_ij, a_ji
            floating loss = 0., bpr_loss = 0., aspect_bpr_loss = 0., del_sqerror, del_bpr, del_aspect_bpr
            floating eps = 1e-9

            floating lr = self.lr
            floating ld_reg = self.lambda_reg
            floating ld_bpr = self.lambda_bpr
            floating lambda_d = self.lambda_d

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
                if has_non_zero(indptr, item_ids, user_ids[i_idx], j_idx):
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

            for _ in prange(num_pair_samples, schedule='guided'):
                idx = rng_pos_pair.generate(thread_id)
                u_idx = p_user_indices[idx]
                i_idx = earlier_indices[idx]
                j_idx = later_indices[idx]
                a_idx = aspect_indices[idx]

                pred = (
                    get_score(G1, n_user_factors, n_item_factors, n_aspect_factors, U, I, A, u_idx, j_idx, a_idx)
                    - get_score(G1, n_user_factors, n_item_factors, n_aspect_factors, U, I, A, u_idx, i_idx, a_idx)
                )

                z = (1.0 / (1.0 + exp(pred)))
                if z < .5:
                    aspect_correct += 1

                del_aspect_bpr = lambda_d * z
                aspect_bpr_loss += log(1 / (1 + exp(-pred)))
                for i in range(n_user_factors):
                    for j in range(n_item_factors):
                        a_ji = I[j_idx, j] - I[i_idx, j]
                        for k in range(n_aspect_factors):
                            del_g1[i, j, k] -= del_aspect_bpr * U[u_idx, i] * a_ji * A[a_idx, k]
                            del_u[u_idx, i] -= del_aspect_bpr * G1[i, j, k] * a_ji * A[a_idx, k]
                            del_i[j_idx, j] -= del_aspect_bpr * G1[i, j, k] * U[u_idx, i] * A[a_idx, k]
                            del_i[i_idx, j] += del_aspect_bpr * G1[i, j, k] * U[u_idx, i] * A[a_idx, k]
                            del_a[a_idx, k] -= del_aspect_bpr * G1[i, j, k] * U[u_idx, i] * a_ji

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
