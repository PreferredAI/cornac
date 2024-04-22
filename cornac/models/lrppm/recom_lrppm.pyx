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
from ..mter.recom_mter cimport get_key


cdef extern from "../bpr/recom_bpr.h" namespace "recom_bpr" nogil:
    cdef int get_thread_num()

cdef int get_key3(int i_id, int j_id, int k_id) noexcept nogil:
    return get_key(get_key(i_id, j_id), k_id)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef floating get_score(floating[:, :] U, floating[:, :] I, floating[:, :] UA, floating[:, :] IA,
                        int n_factors, int u_idx, int i_idx, int a_idx) noexcept nogil:
    cdef floating score = 0.
    cdef int k
    for k in range(n_factors):
        score = score + U[u_idx, k] * UA[a_idx, k] + I[i_idx, k] * IA[a_idx, k] + U[u_idx, k] * I[i_idx, k]
    return score


class LRPPM(Recommender):
    """Learn to Rank user Preferences based on Phrase-level sentiment analysis across Multiple categories (LRPPM)

    Parameters
    ----------
    name: string, optional, default: 'LRPPM'
        The name of the recommender model.

    rating_scale: float, optional, default: 5.0
        The maximum rating score of the dataset.

    n_factors: int, optional, default: 8
        The dimension of the latent factors.

    ld: float, optional, default: 1.0
        The control factor for aspect ranking objective.

    lambda_reg: float, optional, default: 0.01
        The regularization parameter.

    n_top_aspects: int, optional, default: 100
        The number of top scored aspects for each (user, item) pair to construct ranking score.

    alpha: float, optional, default: 0.5
        Trade-off factor for constructing ranking score.

    n_ranking_samples: int, optional, default: 1000
        The number of samples from ranking pairs.

    n_samples: int, optional, default: 200
        The number of samples from all ratings in each iteration.

    max_iter: int, optional, default: 200000
        Maximum number of iterations for training.

    lr: float, optional, default: 0.1
        The learning rate for optimization

    n_threads: int, optional, default: 0
        Number of parallel threads for training. If n_threads=0, all CPU cores will be utilized.
        If seed is not None, n_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U, I, UA, and IA are not None).

    n_threads: int, optional, default: 0
        Number of parallel threads for training. If n_threads=0, all CPU cores will be utilized.
        If seed is not None, n_threads=1 to remove randomness from parallelization.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U, I, A, O, G1, G2, and G3 are not None).

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'I':I, 'UA':UA, 'IA':IA}

        U: ndarray, shape (n_users, n_factors)
            The user latent factors, optional initialization via init_params
            
        I: ndarray, shape (n_users, n_factors)
            The item latent factors, optional initialization via init_params
        
        UA: ndarray, shape (num_aspects, n_factors)
            The user-aspect latent factors, optional initialization via init_params
        
        IA: ndarray, shape (num_aspects, n_factors)
            The item-aspect latent factors, optional initialization via init_params

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    Xu Chen, Zheng Qin, Yongfeng Zhang, Tao Xu. 2016. \
    Learning to Rank Features for Recommendation over Multiple Categories. \
    Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval (SIGIR '16). \
    ACM, New York, NY, USA,  305-314. DOI: https://doi.org/10.1145/2911451.2911549
    """
    def __init__(
        self,
        name="LRPPM",
        rating_scale=5,
        n_factors=8,
        ld=1,
        reg=0.01,
        alpha=1,
        num_top_aspects=99999,
        n_ranking_samples=1000,
        n_samples=200,
        max_iter=200000,
        lr=0.1,
        n_threads=0,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.n_factors = n_factors
        self.rating_scale = rating_scale
        self.ld = ld
        self.reg = reg
        self.alpha = alpha
        self.num_top_aspects = num_top_aspects
        self.n_samples = n_samples
        self.n_ranking_samples = n_ranking_samples
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
        self.U = self.init_params.get("U", None)
        self.I = self.init_params.get("I", None)
        self.UA = self.init_params.get("UA", None)
        self.IA = self.init_params.get("IA", None)

    def _init(self, train_set):
        n_users, n_items = train_set.num_users, train_set.num_items
        n_aspects, n_opinions = train_set.sentiment.num_aspects, train_set.sentiment.num_opinions
        self.num_aspects = n_aspects

        if self.U is None:
            U_shape = (n_users, self.n_factors)
            self.U = uniform(U_shape, random_state=self.rng)
        if self.I is None:
            I_shape = (n_items, self.n_factors)
            self.I = uniform(I_shape, random_state=self.rng)
        if self.UA is None:
            UA_shape = (n_aspects, self.n_factors)
            self.UA = uniform(UA_shape, random_state=self.rng)
        if self.IA is None:
            IA_shape = (n_aspects, self.n_factors)
            self.IA = uniform(UA_shape, random_state=self.rng)

    def _compute_quality_score(self, total_sentiment):
        return 1.0 / (1.0 + np.exp(-total_sentiment))

    def _build_data(self, data_set):
        from time import time

        start_time = time()
        if self.verbose:
            print("Building data started!")

        sentiment = data_set.sentiment
        (u_indices, i_indices, r_values) = data_set.uir_tuple
        keys = np.array([get_key(u, i) for u, i in zip(u_indices, i_indices)], dtype=np.intp)
        cdef IntFloatDict rating_dict = IntFloatDict(keys, np.array(r_values, dtype=np.float64))

        item_aspect_quality = {}
        user_item_aspect = {}
        for uid, sentiment_tup_ids_by_item in sentiment.user_sentiment.items():
            if not self.knows_user(uid):
                continue
            for iid, tup_idx in sentiment_tup_ids_by_item.items():
                for aid, oid, polarity in sentiment.sentiment[tup_idx]:
                    user_item_aspect[(uid, iid, aid)] = user_item_aspect.get((uid, iid, aid), 0) + polarity
                item_aspect_quality[(iid, aid)] = item_aspect_quality.get((iid, aid), 0) + polarity
        i_indices = []
        a_indices = []
        quality_scores = []
        for (iid, aid), total_sentiment in item_aspect_quality.items():
            i_indices.append(iid)
            a_indices.append(aid)
            quality_scores.append(self._compute_quality_score(total_sentiment))
        item_aspect_quality = sp.csr_matrix((quality_scores, (i_indices, a_indices)), shape=(data_set.num_items, data_set.sentiment.num_aspects))
        user_item_aspect_keys = []
        user_item_aspect_scores = []
        for key, value in user_item_aspect.items():
            uid, iid, aid = key
            user_item_aspect_keys.append(get_key3(uid, iid, aid))
            user_item_aspect_scores.append(value)
        user_item_aspect_dict = IntFloatDict(np.array(user_item_aspect_keys, dtype=np.intp), np.array(user_item_aspect_scores, dtype=np.float64))

        user_item_num_aspects = {}
        for (uid, iid, aid) in user_item_aspect.keys():
            user_item_num_aspects[(uid, iid)] = user_item_num_aspects.get((uid, iid), 0) + 1

        if self.verbose:
            total_time = time() - start_time
            print("Building data completed in %d s" % total_time)
        return (
            rating_dict,
            user_item_aspect,
            user_item_aspect_dict,
            item_aspect_quality,
            user_item_num_aspects
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

        self._init(train_set)

        (
            rating_dict,
            user_item_aspect,
            user_item_aspect_dict,
            self.item_aspect_quality, # needed for ranking evaluation (see Eq. 6 in paper)
            user_item_num_aspects
        ) = self._build_data(train_set)
 
        if not self.trainable:
            return self

        X_uids, X_iids, X_aids, X_l_ui = [], [], [], []

        for (uid, iid, aid) in user_item_aspect.keys():
            X_uids.append(uid)
            X_iids.append(iid)
            X_aids.append(aid)
            ui_aspect_cnt = user_item_num_aspects[(uid, iid)]
            ui_neg_aspect_cnt = train_set.sentiment.num_aspects - ui_aspect_cnt
            X_l_ui.append(1.0 / (ui_aspect_cnt * ui_neg_aspect_cnt))

        X_uids = np.array(X_uids, dtype=np.int32)
        X_iids = np.array(X_iids, dtype=np.int32)
        X_aids = np.array(X_aids, dtype=np.int32)
        X_l_ui = np.array(X_l_ui, dtype=np.float32)
        (u_indices, i_indices, r_values) = train_set.uir_tuple

        cdef:
            int n_threads = self.n_threads
            RNGVector rng_pos = RNGVector(n_threads, len(r_values) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_pos_uia = RNGVector(n_threads, len(user_item_aspect) - 1, self.rng.randint(2 ** 31))
            RNGVector rng_neg_uia = RNGVector(n_threads, train_set.sentiment.num_aspects - 1, self.rng.randint(2 ** 31))

        del_u = np.zeros_like(self.U).astype(np.float32)
        del_i = np.zeros_like(self.I).astype(np.float32)
        del_ua = np.zeros_like(self.UA).astype(np.float32)
        del_ia = np.zeros_like(self.IA).astype(np.float32)

        with trange(self.max_iter, disable=not self.verbose) as progress:
            prev_U, prev_I, prev_UA, prev_IA = self.U.copy(), self.I.copy(), self.UA.copy(), self.IA.copy()
            for epoch in progress:
                correct, skipped, loss, ranking_loss, r_loss = self._fit(
                    rng_pos, rng_pos_uia, rng_neg_uia,
                    n_threads,
                    u_indices.astype(np.int32), i_indices.astype(np.int32), r_values.astype(np.float32),
                    X_uids, X_iids, X_aids, X_l_ui,
                    rating_dict,
                    user_item_aspect_dict,
                    self.U, self.I, self.UA, self.IA,
                    del_u, del_i, del_ua, del_ia,
                )

                progress.set_postfix({
                    "loss": "%.2f" % (loss / self.n_samples),
                    "ranking_loss": "%.2f" % (ranking_loss / (self.n_ranking_samples - skipped)),
                    "r_loss": "%.2f" % (r_loss / (self.n_ranking_samples - skipped)),
                    "correct": "%.2f%%" % (100.0 * correct / (self.n_ranking_samples - skipped)),
                    "skipped": "%.2f%%" % (100.0 * skipped / self.n_ranking_samples)
                })
                if (
                    np.all(np.isclose(self.U, prev_U))
                    and np.all(np.isclose(self.I, prev_I))
                    and np.all(np.isclose(self.UA, prev_UA))
                    and np.all(np.isclose(self.IA, prev_IA))
                ):
                    print('Stop training because model converged!')
                    break
                else:
                    prev_U, prev_I, prev_UA, prev_IA = self.U.copy(), self.I.copy(), self.UA.copy(), self.IA.copy()

        if self.verbose:
            print('Optimization finished!')

        return self

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit(
        self,
        RNGVector rng_pos,
        RNGVector rng_pos_uia,
        RNGVector rng_neg_uia,
        int n_threads,
        integral[:] u_indices, integral[:] i_indices, floating[:] r_values,
        integral[:] X_uids,
        integral[:] X_iids,
        integral[:] X_aids,
        floating[:] X_l_ui,
        IntFloatDict rating_dict,
        IntFloatDict user_item_aspect_dict,
        floating[:, :] U,
        floating[:, :] I,
        floating[:, :] UA,
        floating[:, :] IA,
        np.ndarray[np.float32_t, ndim=2] del_u,
        np.ndarray[np.float32_t, ndim=2] del_i,
        np.ndarray[np.float32_t, ndim=2] del_ua,
        np.ndarray[np.float32_t, ndim=2] del_ia):
        """Fit the model parameters (U, I, UA, IA)
        """
        cdef:
            long s, i_index, j_index, correct = 0, skipped = 0
            long n_users = self.num_users
            long n_items = self.num_items
            long n_aspects = self.num_aspects
            long n_factors = self.n_factors
            int num_samples = self.n_samples
            int num_ranking_samples = self.n_ranking_samples

            integral _, i, j, k, idx, u_idx, i_idx, a_idx, a_jdx, thread_id, ui_key
            floating z, score, pred, r_pred, l_ui
            floating loss = 0., ranking_loss = 0., r_loss = 0., del_rating, del_ranking, del_sqerror
            floating lr = self.lr
            floating reg = self.reg
            floating ld = self.ld

        del_u.fill(0)
        del_i.fill(0)
        del_ua.fill(0)
        del_ia.fill(0)
        with nogil, parallel(num_threads=n_threads):
            thread_id = get_thread_num()
            for _ in prange(num_samples, schedule='guided'):
                idx = rng_pos.generate(thread_id)
                u_idx = u_indices[idx]
                i_idx = i_indices[idx]
                score = r_values[idx]
                r_pred = 0.
                for k in range(n_factors):
                    r_pred = r_pred + U[u_idx, k] * I[i_idx, k]
                del_sqerror = 2 * (r_pred - score)
                loss += (score - r_pred) * (score - r_pred)
                for k in range(n_factors):
                    del_u[u_idx, k] += del_sqerror * I[i_idx, k]
                    del_i[i_idx, k] += del_sqerror * U[u_idx, k]

            for _ in prange(num_ranking_samples, schedule='guided'):
                idx = rng_pos_uia.generate(thread_id)
                u_idx = X_uids[idx]
                i_idx = X_iids[idx]
                a_idx = X_aids[idx]
                a_jdx = rng_neg_uia.generate(thread_id)
                if user_item_aspect_dict.my_map.find(get_key3(u_idx, i_idx, a_jdx)) != user_item_aspect_dict.my_map.end():
                    skipped += 1
                    continue

                # ranking
                pred = (
                    get_score(U, I, UA, IA, n_factors, u_idx, i_idx, a_idx)
                    - get_score(U, I, UA, IA, n_factors, u_idx, i_idx, a_jdx)
                )
                z = (1.0 / (1.0 + exp(pred)))
                if z < .5:
                    correct += 1
                del_ranking = ld * z
                ranking_loss += ld * log(1.0 / (1.0 + exp(-pred)))
                for k in range(n_factors):
                    del_u[u_idx, k] -= del_ranking * (UA[a_idx, k] - UA[a_jdx, k])
                    del_i[i_idx, k] -= del_ranking * (IA[a_idx, k] - IA[a_jdx, k])
                    del_ua[a_idx, k] -= del_ranking * U[u_idx, k]
                    del_ua[a_jdx, k] += del_ranking * U[u_idx, k]
                    del_ia[a_idx, k] -= del_ranking * I[i_idx, k]
                    del_ia[a_jdx, k] += del_ranking * I[i_idx, k]

                # rating
                r_pred = 0.
                for k in range(n_factors):
                    r_pred = r_pred + U[u_idx, k] * I[i_idx, k]
                score = rating_dict.my_map[get_key(u_idx, i_idx)]
                l_ui = X_l_ui[idx]
                del_rating = 2 * l_ui * (score - r_pred)
                r_loss += l_ui * (score - r_pred) * (score - r_pred)
                for k in range(n_factors):
                    del_u[u_idx, k] += del_rating * I[i_idx, k]
                    del_i[i_idx, k] += del_rating * U[u_idx, k]


            # Update using sgd and ensure non-negative constraints
            for i in range(n_factors):
                for j in range(n_users):
                    if del_u[j, i] != 0:
                        del_u[j, i] += reg * U[j, i]
                    U[j, i] -= lr * del_u[j, i]
                    if U[j, i] < 0:
                        U[j, i] = 0
                for j in range(n_items):
                    if del_i[j, i] != 0:
                        del_i[j, i] += reg * I[j, i]
                    I[j, i] -= lr * del_i[j, i]
                    if I[j, i] < 0:
                        I[j, i] = 0
                for j in range(n_aspects):
                    if del_ua[j, i] != 0:
                        del_ua[j, i] += reg * UA[j, i]
                    UA[j, i] -= lr * del_ua[j, i]
                    if UA[j, i] < 0:
                        UA[j, i] = 0
                    if del_ia[j, i] != 0:
                        del_ia[j, i] += reg * IA[j, i]
                    IA[j, i] -= lr * del_ia[j, i]
                    if IA[j, i] < 0:
                        IA[j, i] = 0

        return correct, skipped, loss, ranking_loss, r_loss

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
            if not self.knows_user(u_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d" & u_idx
                )
            
            item_scores = self.I.dot(self.U[u_idx])
            return item_scores
        else:
            if not (self.knows_user(u_idx) and self.knows_item(i_idx)):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (u_idx, i_idx)
                )
            item_score = self.I[i_idx].dot(self.U[u_idx])
            return item_score

    def rank(self, user_idx, item_indices=None, k=-1):
        if self.alpha > 0 and self.num_top_aspects > 0:
            n_items = self.num_items
            num_top_aspects = min(self.num_top_aspects, self.num_aspects)
            item_aspect_scores = self.UA.dot(self.U[user_idx]) + self.I.dot(self.IA.T) + np.expand_dims(self.I.dot(self.U[user_idx]), axis=1)
            top_aspect_ids = (-item_aspect_scores).argsort(axis=1)[:, :num_top_aspects]
            iids = np.repeat(range(n_items), num_top_aspects).reshape(n_items, num_top_aspects)
            top_aspect_scores = item_aspect_scores[iids, top_aspect_ids]
            known_item_scores = (
                self.alpha * (top_aspect_scores * self.item_aspect_quality[iids, top_aspect_ids].A).mean(axis=1) * self.rating_scale
                + (1 - self.alpha) * self.I.dot(self.U[user_idx])
            )

            # check if the returned scores also cover unknown items
            # if not, all unknown items will be given the MIN score
            if len(known_item_scores) == self.total_items:
                all_item_scores = known_item_scores
            else:
                all_item_scores = np.ones(self.total_items) * np.min(
                    known_item_scores
                )
                all_item_scores[: self.num_items] = known_item_scores

            # rank items based on their scores
            item_indices = (
                np.arange(self.num_items)
                if item_indices is None
                else np.asarray(item_indices)
            )
            item_scores = all_item_scores[item_indices]

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