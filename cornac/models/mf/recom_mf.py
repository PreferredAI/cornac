# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np
import scipy.sparse as sp
from cornac.models.recommender import Recommender
from cornac.utils.generic_utils import intersects
from cornac.exception import ScoreException
import mf


class MF(Recommender):
    """Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.01
        The learning rate.

    lambda_reg: float, optional, default: 0.001
        The lambda value used for regularization.

    use_bias: boolean, optional, default: True
        When True, user, item, and global biases are used.

    early_stop: boolean, optional, default: False
        When True, delta loss will be checked after each iteration to stop learning earlier.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    References
    ----------
    * Koren, Y., Bell, R., & Volinsky, C. Matrix factorization techniques for recommender systems. \
    In Computer, (8), 30-37. 2009.
    """

    def __init__(self, k=10, max_iter=20, learning_rate=0.01, lambda_reg=0.02, use_bias=True, early_stop=False,
                 verbose=False):
        Recommender.__init__(self, name='MF', verbose=verbose)

        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.use_bias = use_bias
        self.early_stop = early_stop
        self.fitted = False

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

        (rid, cid, val) = sp.find(train_set.matrix)
        data = [(u, i, r) for u, i, r in zip(rid, cid, val)]

        self.u_factors, self.i_factors, self.u_biases, self.i_biases = mf.sgd(data=data,
                                                                              num_users=train_set.num_users,
                                                                              num_items=train_set.num_items,
                                                                              k=self.k,
                                                                              max_iter=self.max_iter,
                                                                              learning_rate=self.learning_rate,
                                                                              lambda_reg=self.lambda_reg,
                                                                              global_mean=train_set.global_mean,
                                                                              use_bias=self.use_bias,
                                                                              early_stop=self.early_stop,
                                                                              verbose=self.verbose)
        self.fitted = True

    def score(self, user_id, item_id):
        """Predict the scores/ratings of a user for a list of items.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score predictions.

        item_id: int, required
            The index of the item to be scored by the user.

        Returns
        -------
        A scalar
            The estimated score (e.g., rating) for the user and item of interest
        """
        if not self.fitted:
            raise ValueError('You need to fit the model first!')

        unk_user = self.train_set.is_unk_user(user_id)
        unk_item = self.train_set.is_unk_item(item_id)

        if self.use_bias:
            score_pred = self.train_set.global_mean
            if not unk_user:
                score_pred += self.u_biases[user_id]
            if not unk_item:
                score_pred += self.i_biases[item_id]

            if not unk_user and not unk_item:
                score_pred += np.dot(self.u_factors[user_id], self.i_factors[item_id])
        else:
            if unk_user or unk_item:
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))

            score_pred = np.dot(self.u_factors[user_id], self.i_factors[item_id])

        return score_pred

    def rank(self, user_id, candidate_item_ids=None):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform item raking.

        candidate_item_ids: 1d array, optional, default: None
            A list of item indices to be ranked by the user.
            If `None`, list of ranked known item indices will be returned

        Returns
        -------
        Numpy 1d array
            Array of item indices sorted (in decreasing order) relative to some user preference scores.
        """
        if not self.fitted:
            raise ValueError('You need to fit the model first!')

        if self.train_set.is_unk_user(user_id):
            if self.use_bias:
                known_item_scores = self.i_biases
            else:
                return self.default_rank(candidate_item_ids)
        else:
            known_item_scores = np.dot(self.i_factors, self.u_factors[user_id])

        if candidate_item_ids is None:
            ranked_item_ids = known_item_scores.argsort()[::-1]
            return ranked_item_ids
        else:
            num_items = max(self.train_set.num_items, max(candidate_item_ids) + 1)
            pref_scores = np.ones(num_items) * self.train_set.min_rating  # use min_rating to shift unk items to the end
            pref_scores[:self.train_set.num_items] = known_item_scores

            ranked_item_ids = pref_scores.argsort()[::-1]
            ranked_item_ids = intersects(ranked_item_ids, candidate_item_ids, assume_unique=True)

            return ranked_item_ids
