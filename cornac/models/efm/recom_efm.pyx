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
from cython.parallel import prange, parallel
from cython cimport floating, integral
from libc.math cimport sqrt
from scipy.linalg.cython_blas cimport sdot, ddot
from ...exception import ScoreException
from ...utils.common import intersects

from ..recommender import Recommender
from collections import Counter, OrderedDict
import scipy.sparse as sp
import numpy as np
cimport numpy as np


cdef floating _dot(int n, floating *x, int incx,
                   floating *y, int incy) nogil:
    if floating is float:
        return sdot(&n, x, &incx, y, &incy)
    else:
        return ddot(&n, x, &incx, y, &incy)


class EFM(Recommender):
    """Explict Factor Models

    Parameters
    ----------
    num_explicit_factors: int, optional, default: 50
        The dimension of the explicit factors.

    num_latent_factors: int, optional, default: 50
        The dimension of the latent factors.

    num_most_cared_aspects: int, optional, default: 15
        The number of most cared aspects for each user.

    rating_scale: float, optional, default: 5.0
        The maximum rating score of the dataset.

    alpha: float, optional, default: 0.85
        Trace off factor for constructing ranking score.

    lambda_x: float, optional, default: 1
        The regularization parameter for user aspect attentions.

    lambda_y: float, optional, default: 1
        The regularization parameter for item aspect qualities.

    lambda_u: float, optional, default: 0.01
        The regularization parameter for user and item explicit factors.

    lambda_h: float, optional, default: 0.01
        The regularization parameter for user and item latent factors.

    lambda_v: float, optional, default: 0.01
        The regularization parameter for V.

    use_item_aspect_popularity: boolean, optional, default: True
        When False, item aspect quality score computation omits item aspect frequency out of its formular.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs.

    name: string, optional, default: 'EFM'
        The name of the recommender model.

    num_threads: int, optional, default: 0
        Number of parallel threads for training.
        If 0, all CPU cores will be utilized.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already 
        pre-trained (U1, U2, V, H1, and H2 are not None).

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U1':U1, 'U2':U2, 'V':V', H1':H1, 'H2':H2}
        U1: ndarray, shape (n_users, n_explicit_factors)
            The user explicit factors, optional initialization via init_params.
        U2: ndarray, shape (n_ratings, n_explicit_factors)
            The item explicit factors, optional initialization via init_params.
        V: ndarray, shape (n_aspects, n_explict_factors)
            The aspect factors, optional initialization via init_params.
        H1: ndarray, shape (n_users, n_latent_factors)
            The user latent factors, optional initialization via init_params.
        H2: ndarray, shape (n_ratings, n_latent_factors)
            The item latent factors, optional initialization via init_params.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    Yongfeng Zhang, Guokun Lai, Min Zhang, Yi Zhang, Yiqun Liu, and Shaoping Ma. 2014.
    Explicit factor models for explainable recommendation based on phrase-level sentiment analysis.
    In Proceedings of the 37th international ACM SIGIR conference on Research & development in information retrieval (SIGIR '14).
    ACM, New York, NY, USA, 83-92. DOI: https://doi.org/10.1145/2600428.2609579
    """

    def __init__(self,  name="EFM",
                 num_explicit_factors=50, num_latent_factors=50, num_most_cared_aspects=15,
                 rating_scale=5.0, alpha=0.85,
                 lambda_x=1, lambda_y=1, lambda_u=0.01, lambda_h=0.01, lambda_v=0.01,
                 use_dominant=False,
                 use_item_aspect_popularity=True, max_iter=100,
                 num_threads=0, trainable=True, verbose=False, init_params=None, seed=None):

        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.num_explicit_factors = num_explicit_factors
        self.num_latent_factors = num_latent_factors
        self.num_most_cared_aspects = num_most_cared_aspects
        self.rating_scale = rating_scale
        self.alpha = alpha
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.lambda_u = lambda_u
        self.lambda_h = lambda_h
        self.lambda_v = lambda_v
        self.use_item_aspect_popularity = use_item_aspect_popularity
        self.max_iter = max_iter
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
        """Fit the model.

        Parameters
        ----------
        train_set: :obj:`cornac.data.MultimodalTrainSet`
            Multimodal training set.

        """
        Recommender.fit(self, train_set)

        from ...utils import get_rng
        from ...utils.init_utils import uniform

        rng = get_rng(self.seed)
        num_factors = self.num_explicit_factors + self.num_latent_factors
        high = np.sqrt(self.rating_scale / num_factors)
        self.U1 = self.init_params.get('U1', uniform((self.train_set.num_users, self.num_explicit_factors),
                                                     high=high, random_state=rng))
        self.U2 = self.init_params.get('U2', uniform((self.train_set.num_items, self.num_explicit_factors),
                                                     high=high, random_state=rng))
        self.V = self.init_params.get('V', uniform((self.train_set.sentiment.num_aspects, self.num_explicit_factors),
                                                   high=high, random_state=rng))
        self.H1 = self.init_params.get('H1', uniform((self.train_set.num_users, self.num_latent_factors),
                                                     high=high, random_state=rng))
        self.H2 = self.init_params.get('H2', uniform((self.train_set.num_items, self.num_latent_factors),
                                                     high=high, random_state=rng))

        if not self.trainable:
            return

        A, X, Y = self._build_matrices(self.train_set)
        A_user_counts = np.ediff1d(A.indptr)
        A_item_counts = np.ediff1d(A.tocsc().indptr)
        A_uids = np.repeat(np.arange(self.train_set.num_users), A_user_counts).astype(A.indices.dtype)
        X_user_counts = np.ediff1d(X.indptr)
        X_aspect_counts = np.ediff1d(X.tocsc().indptr)
        X_uids = np.repeat(np.arange(self.train_set.num_users), X_user_counts).astype(X.indices.dtype)
        Y_item_counts = np.ediff1d(Y.indptr)
        Y_aspect_counts = np.ediff1d(Y.tocsc().indptr)
        Y_iids = np.repeat(np.arange(self.train_set.num_items), Y_item_counts).astype(Y.indices.dtype)
        self._fit_efm(self.num_threads,
                      A.data.astype(np.float32), A_uids, A.indices, A_user_counts, A_item_counts,
                      X.data.astype(np.float32), X_uids, X.indices, X_user_counts, X_aspect_counts,
                      Y.data.astype(np.float32), Y_iids, Y.indices, Y_item_counts, Y_aspect_counts,
                      self.U1, self.U2, self.V, self.H1, self.H2)

    def get_params(self):
        """Get model parameters in the form of dictionary including matrices: U1, U2, V, H1, H2
        """
        return {
            'U1': self.U1,
            'U2': self.U2,
            'V': self.V,
            'H1': self.H1,
            'H2': self.H2,
        }

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _fit_efm(self, int num_threads,
                 floating[:] A, integral[:] A_uids, integral[:] A_iids, integral[:] A_user_counts, integral[:] A_item_counts,
                 floating[:] X, integral[:] X_uids, integral[:] X_aids, integral[:] X_user_counts, integral[:] X_aspect_counts,
                 floating[:] Y, integral[:] Y_iids, integral[:] Y_aids, integral[:] Y_item_counts, integral[:] Y_aspect_counts,
                 floating[:, :] U1, floating[:, :] U2, floating[:, :] V, floating[:, :] H1, floating[:, :] H2):
        """Fit the model parameters (U1, U2, V, H1, H2)
        """
        cdef:
            long num_users = self.train_set.num_users
            long num_items = self.train_set.num_items
            long num_aspects = self.train_set.sentiment.num_aspects
            int num_explicit_factors = self.num_explicit_factors
            int num_latent_factors = self.num_latent_factors

            floating lambda_x = self.lambda_x
            floating lambda_y = self.lambda_y
            floating lambda_u = self.lambda_u
            floating lambda_h = self.lambda_h
            floating lambda_v = self.lambda_v

            floating prediction, score, loss

            np.ndarray[np.float32_t, ndim=2] U1_numerator = np.empty((num_users, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] U1_denominator = np.empty((num_users, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] U2_numerator = np.empty((num_items, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] U2_denominator = np.empty((num_items, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] V_numerator = np.empty((num_aspects, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] V_denominator = np.empty((num_aspects, num_explicit_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] H1_numerator = np.empty((num_users, num_latent_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] H1_denominator = np.empty((num_users, num_latent_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] H2_numerator = np.empty((num_items, num_latent_factors), dtype=np.float32)
            np.ndarray[np.float32_t, ndim=2] H2_denominator = np.empty((num_items, num_latent_factors), dtype=np.float32)
            int i, j, k, idx
            long n_ratings

            floating eps = 1e-9

        for t in range(1, self.max_iter + 1):

            loss = 0.
            U1_numerator.fill(0)
            U1_denominator.fill(0)
            U2_numerator.fill(0)
            U2_denominator.fill(0)
            V_numerator.fill(0)
            V_denominator.fill(0)
            H1_numerator.fill(0)
            H1_denominator.fill(0)
            H2_numerator.fill(0)
            H2_denominator.fill(0)

            with nogil, parallel(num_threads=num_threads):
                # compute numerators and denominators for all factors
                for idx in prange(A.shape[0]):
                    i = A_uids[idx]
                    j = A_iids[idx]
                    prediction = _dot(num_explicit_factors, &U1[i, 0], 1, &U2[j, 0], 1) \
                                 + _dot(num_latent_factors, &H1[i, 0], 1, &H2[j, 0], 1)
                    score = A[idx]
                    loss += (prediction - score) * (prediction - score)
                    for k in range(num_explicit_factors):
                        U1_numerator[i, k] += score * U2[j, k]
                        U1_denominator[i, k] += prediction * U2[j, k]
                        U2_numerator[j, k] += score * U1[i, k]
                        U2_denominator[j, k] += prediction * U1[i, k]

                    for k in range(num_latent_factors):
                        H1_numerator[i, k] += score * H2[j, k]
                        H1_denominator[i, k] += prediction * H2[j, k]
                        H2_numerator[j, k] += score * H1[i, k]
                        H2_denominator[j, k] += prediction * H1[i, k]

                for idx in prange(X.shape[0]):
                    i = X_uids[idx]
                    j = X_aids[idx]
                    prediction = _dot(num_explicit_factors, &U1[i, 0], 1, &V[j, 0], 1)
                    score = X[idx]
                    loss += (prediction - score) * (prediction - score)
                    for k in range(num_explicit_factors):
                        V_numerator[j, k] += lambda_x * score * U1[i, k]
                        V_denominator[j, k] += lambda_x * prediction * U1[i, k]
                        U1_numerator[i, k] += lambda_x * score * V[j, k]
                        U1_denominator[i, k] += lambda_x * prediction * V[j, k]

                for idx in prange(Y.shape[0]):
                    i = Y_iids[idx]
                    j = Y_aids[idx]
                    prediction = _dot(num_explicit_factors, &U2[i, 0], 1, &V[j, 0], 1)
                    score = Y[idx]
                    loss += (prediction - score) * (prediction - score)
                    for k in range(num_latent_factors):
                        V_numerator[j, k] += lambda_y * score * U2[i, k]
                        V_denominator[j, k] += lambda_y * prediction * U2[i, k]
                        U2_numerator[i, k] += lambda_y * score * V[j, k]
                        U2_denominator[i, k] += lambda_y * prediction * V[j, k]

                # update V
                for i in prange(num_aspects):
                    for j in range(num_explicit_factors):
                        loss += lambda_v * V[i, j] * V[i, j]
                        V_denominator[i, j] += (X_aspect_counts[i] + Y_aspect_counts[i]) * lambda_v * V[i, j] + eps
                        V[i, j] *= sqrt(V_numerator[i, j] / V_denominator[i, j])

                # update U1, H1
                for i in prange(num_users):
                    for j in range(num_explicit_factors):
                        loss += lambda_u * U1[i, j] * U1[i, j]
                        U1_denominator[i, j] += (A_user_counts[i] + X_user_counts[i])* lambda_u * U1[i, j] + eps
                        U1[i, j] *= sqrt(U1_numerator[i, j] / U1_denominator[i, j])
                    for j in range(num_latent_factors):
                        loss += lambda_h * H1[i, j] * H1[i, j]
                        H1_denominator[i, j] += A_user_counts[i] * lambda_h * H1[i, j] + eps
                        H1[i, j] *= sqrt(H1_numerator[i, j] / H1_denominator[i, j])

                # update U2, H2
                for i in prange(num_items):
                    for j in range(num_explicit_factors):
                        loss += lambda_u * U2[i, j] * U2[i, j]
                        U2_denominator[i, j] += (A_item_counts[i] + Y_item_counts[i]) * lambda_u * U2[i, j] + eps
                        U2[i, j] *= sqrt(U2_numerator[i, j] / U2_denominator[i, j])
                    for j in range(num_latent_factors):
                        loss += lambda_h * H2[i, j] * H2[i, j]
                        H2_denominator[i, j] += A_item_counts[i] * lambda_h * H2[i, j] + eps
                        H2[i, j] *= sqrt(H2_numerator[i, j] / H2_denominator[i, j])

            if self.verbose:
                print('iter: %d, loss: %f' % (t, loss))

        if self.verbose:
            print('Optimization finished!')

    def _build_matrices(self, train_set):
        sentiment = train_set.sentiment
        ratings = []
        map_uid = []
        map_iid = []

        for uid, iid, rating in self.train_set.uir_iter():
            if train_set.is_unk_user(uid) or train_set.is_unk_item(iid):
                continue
            ratings.append(rating)
            map_uid.append(uid)
            map_iid.append(iid)

        ratings = np.asarray(ratings, dtype=np.float).flatten()
        map_uid = np.asarray(map_uid, dtype=np.int).flatten()
        map_iid = np.asarray(map_iid, dtype=np.int).flatten()
        A = sp.csr_matrix((ratings, (map_uid, map_iid)),
                          shape=(train_set.num_users, train_set.num_items))

        attention_scores = []
        map_uid = []
        map_aspect_id = []
        for uid, sentiment_tup_ids_by_item in sentiment.user_sentiment.items():
            if train_set.is_unk_user(uid):
                continue
            user_aspects = [tup[0] for tup_id in sentiment_tup_ids_by_item.values()
                                   for tup in sentiment.sentiment[tup_id]]
            user_aspect_count = Counter(user_aspects)
            for aid, count in user_aspect_count.items():
                attention_scores.append(self._compute_attention_score(count))
                map_uid.append(uid)
                map_aspect_id.append(aid)

        attention_scores = np.asarray(attention_scores, dtype=np.float).flatten()
        map_uid = np.asarray(map_uid, dtype=np.int).flatten()
        map_aspect_id = np.asarray(map_aspect_id, dtype=np.int).flatten()
        X = sp.csr_matrix((attention_scores, (map_uid, map_aspect_id)),
                          shape=(train_set.num_users, sentiment.num_aspects))

        quality_scores = []
        map_iid = []
        map_aspect_id = []

        for iid, sentiment_tup_ids_by_user in sentiment.item_sentiment.items():
            if train_set.is_unk_item(iid):
                continue
            item_aspects = [tup[0] for tup_id in sentiment_tup_ids_by_user.values()
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

        quality_scores = np.asarray(quality_scores, dtype=np.float).flatten()
        map_iid = np.asarray(map_iid, dtype=np.int).flatten()
        map_aspect_id = np.asarray(map_aspect_id, dtype=np.int).flatten()
        Y = sp.csr_matrix((quality_scores, (map_iid, map_aspect_id)),
                          shape=(train_set.num_items, sentiment.num_aspects))

        if self.verbose:
            print('Building matrices completed!')

        return A, X, Y

    def _compute_attention_score(self, count):
        return 1 + (self.rating_scale - 1) * (2 / (1 + np.exp(-count)) - 1)

    def _compute_quality_score(self, sentiment):
        return 1 + (self.rating_scale - 1) / (1 + np.exp(-sentiment))

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
        if item_id is None:
            if self.train_set.is_unk_user(user_id):
                raise ScoreException("Can't make score prediction for (user_id=%d" & user_id)
            item_scores = self.U2.dot(self.U1[user_id, :]) + self.H2.dot(self.H1[user_id, :])
            return item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
            item_score = self.U2[item_id, :].dot(self.U1[user_id, :]) + self.H2[item_id, :].dot(self.H1[user_id, :])
            return item_score

    def rank(self, user_id, item_ids=None):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform item raking.

        item_ids: 1d array, optional, default: None
            A list of candidate item indices to be ranked by the user.
            If `None`, list of ranked known item indices and their scores will be returned

        Returns
        -------
        Tuple of `item_rank`, and `item_scores`. The order of values
        in item_scores are corresponding to the order of their ids in item_ids

        """
        X_ = self.U1[user_id, :].dot(self.V.T)
        most_cared_aspects_indices = (-X_).argsort()[:self.num_most_cared_aspects]
        most_cared_X_ = X_[most_cared_aspects_indices]
        most_cared_Y_ = self.U2.dot(self.V[most_cared_aspects_indices, :].T)
        explicit_scores = most_cared_X_.dot(most_cared_Y_.T) / (self.num_most_cared_aspects * self.rating_scale)
        item_scores = self.alpha * explicit_scores + (1 - self.alpha) * self.score(user_id)

        if item_ids is None:
            item_scores = item_scores
            item_rank = item_scores.argsort()[::-1]
        else:
            num_items = max(self.train_set.num_items, max(item_ids) + 1)
            item_scores = np.ones(num_items) * np.min(item_scores)
            item_scores[:self.train_set.num_items] = item_scores
            item_rank = item_scores.argsort()[::-1]
            item_rank = intersects(item_rank, item_ids, assume_unique=True)
            item_scores = item_scores[item_ids]
        return item_rank, item_scores
