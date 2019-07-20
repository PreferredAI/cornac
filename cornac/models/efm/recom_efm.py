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

from ...exception import ScoreException
from ..recommender import Recommender
from collections import Counter, OrderedDict
import numpy as np

class EFM(Recommender):
    """Explict Factor Models

    Parameters
    ----------
    num_explicit_factors: int, optional, default: 50
        The dimension of the explicit factors.

    num_latent_factors: int, optional, default: 50
        The dimension of the latent factors.

    num_most_cared_aspects: int, optional, default:15
        The number of most cared aspects for each user.

    rating_scale: float, optional, default: 5.0
        The maximum rating score of the dataset.

    alpha: float, optional, default:0.85
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
 
    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.01
        The learning rate for AdamOptimizer.

    name: string, optional, default: 'CDR'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already 
        pre-trained (U and V are not None).

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U1':U1, 'U2':U2, 'V':V', H1':H1, 'H2':H2}
        U1: ndarray, shape (n_users, n_explicit_factors)
            The user explicit factors, optional initialization via init_params.
        U2: ndarray, shape (n_items, n_explicit_factors)
            The item explicit factors, optional initialization via init_params.
        V: ndarray, shape (n_aspects, n_explict_factors)
            The aspect factors, optional initialization via init_params.
        H1: ndarray, shape (n_users, n_latent_factors)
            The user latent factors, optional initialization via init_params.
        H2: ndarray, shape (n_items, n_latent_factors)
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
                 max_iter=100, learning_rate=0.01,
                 trainable=True, verbose=False, init_params=None, seed=None):

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
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.init_params = {} if init_params is None else init_params
        self.seed = seed

    def fit(self, train_set):
        Recommender.fit(self, train_set)

        from ...utils import get_rng
        from ...utils.init_utils import uniform

        rng = get_rng(self.seed)
        num_factors = self.num_explicit_factors + self.num_latent_factors
        low = np.sqrt(1. / num_factors)
        high = np.sqrt(5. / num_factors)
        self.U1 = self.init_params.get('U1', uniform((self.train_set.num_users, self.num_explicit_factors), low=low, high=high, random_state=rng))
        self.U2 = self.init_params.get('U2', uniform((self.train_set.num_items, self.num_explicit_factors), low=low, high=high, random_state=rng))
        self.V = self.init_params.get('V', uniform((self.train_set.sentiment.num_aspects, self.num_explicit_factors), low=low, high=high, random_state=rng))
        self.H1 = self.init_params.get('H1', uniform((self.train_set.num_users, self.num_latent_factors), low=low, high=high, random_state=rng))
        self.H2 = self.init_params.get('H2', uniform((self.train_set.num_items, self.num_latent_factors), low=low, high=high, random_state=rng))

        if self.trainable:
            self._fit_efm()

    def get_params(self):
        """Get model parameters in the form of dictionary including matrices: U1, U2, V, H1, H2
        """
        return {
            'U1': self.U1,
            'U2': self.U2,
            'V': self.V,
            'H1': self.H1,
            'H2': self.H2
        }

    def _fit_efm(self):
        from .efm import sgd_efm

        A, X, Y = self._build_matrices(self.train_set)
        self.U1, self.U2, self.V, self.H1, self.H2 = sgd_efm(A, X, Y, self.U1, self.U2, self.V, self.H1, self.H2,
                                                             self.num_explicit_factors, self.num_latent_factors,
                                                             self.lambda_x, self.lambda_y, self.lambda_u, self.lambda_h, self.lambda_v,
                                                             self.max_iter, self.learning_rate, self.verbose)

        if self.verbose:
            print('Learning completed!')

    def _build_matrices(self, train_set):
        sentiment = train_set.sentiment
        A = np.zeros((train_set.num_users, train_set.num_items))
        X = np.zeros((train_set.num_users, sentiment.num_aspects))
        Y = np.zeros((train_set.num_items, sentiment.num_aspects))

        for uid, iid, rating in self.train_set.uir_iter():
            if train_set.is_unk_user(uid) or train_set.is_unk_item(iid):
                continue
            A[uid, iid] = rating

        for uid, sentiment_tuple_ids_by_item in sentiment.user_sentiment.items():
            if train_set.is_unk_user(uid):
                continue
            user_aspects = [tup[0] for tup_id in sentiment_tuple_ids_by_item.values()
                                   for tup in sentiment.sentiment[tup_id]]
            user_aspect_count = Counter(user_aspects)
            for aid, count in user_aspect_count.items():
                X[uid, aid] = self._compute_attention_score(count)

        for iid, sentiment_tuple_ids_by_user in sentiment.item_sentiment.items():
            if train_set.is_unk_item(iid):
                continue
            total_sentiment_by_aspect = OrderedDict()
            for tup_id in sentiment_tuple_ids_by_user.values():
                for aid, _, sentiment_polarity in sentiment.sentiment[tup_id]:
                    total_sentiment_by_aspect[aid] = total_sentiment_by_aspect.get(aid, 0) + sentiment_polarity
            for aid, total_sentiment in total_sentiment_by_aspect.items():
                Y[iid, aid] = self._compute_quality_score(total_sentiment)
        return A, X, Y

    def _compute_attention_score(self, count):
        return 1 + (self.rating_scale - 1) * (2 / (1 + np.exp(-count)) - 1)

    def _compute_quality_score(self, sentiment):
        return 1 + (self.rating_scale - 1) / (1 + np.exp(-sentiment))

    def score(self, user_id, item_id=None):
        if item_id is None:
            if self.train_set.is_unk_user(user_id):
                raise ScoreException("Can't make score prediction for (user_id=%d" & user_id)

            known_item_scores = self.U2.dot(self.U1[user_id, :]) + self.H2.dot(self.H1[user_id, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
            user_pred = self.U2[item_id, :].dot(self.U1[user_id, :]) + self.H2[item_id, :].dot(self.H1[user_id, :])
            return user_pred

    def rank(self, user_id, item_ids=None):
        X_ = self.U1[user_id, :].dot(self.V.T)
        Y_ = self.U2.dot(self.V.T)
        most_cared_aspects_indices = (-X_).argsort()[:self.num_most_cared_aspects]
        num_items = Y_.shape[0]
        most_cared_X_ = X_[most_cared_aspects_indices]
        most_cared_Y_ = Y_[np.repeat(np.arange(num_items), self.num_most_cared_aspects).reshape(num_items, self.num_most_cared_aspects), most_cared_aspects_indices]
        explicit_scores = most_cared_X_.dot(most_cared_Y_.T) / (self.num_most_cared_aspects * self.rating_scale)
        known_item_scores = self.alpha * explicit_scores + (1 - self.alpha) * self.score(user_id)

        if item_ids is None:
            item_scores = known_item_scores
            item_rank = item_scores.argsort()[::-1]
        else:
            num_items = max(self.train_set.num_items, max(item_ids) + 1)
            item_scores = np.ones(num_items) * np.min(known_item_scores)
            item_scores[:self.train_set.num_items] = known_item_scores
            item_rank = item_scores.argsort()[::-1]
            item_rank = intersects(item_rank, item_ids, assume_unique=True)
            item_scores = item_scores[item_ids]
        return item_rank, item_scores