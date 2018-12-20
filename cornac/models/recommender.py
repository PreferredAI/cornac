# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:18:14 2017
@author:    Aghiles Salah
            Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..exception import ScoreException
from ..utils.generic_utils import intersects, excepts, clipping
import numpy as np

class Recommender:
    """Generic class for a recommender model. All recommendation models should inherit from this class 
    
    Input Parameters
    ----------------
    name: char, required
        The name of the recommender model

    trainable:boolean, optional, default: True
        When False, the model is not trained

    Other attributes
    ----------------
    perfomance: dictionary, optional, default: None
        A collection of recommender models
    """

    def __init__(self, name, trainable=True, verbose = False):
        self.name = name
        self.trainable = trainable

        self.train_set = None
        self.verbose = verbose
        self.perfomance = None


    def fit(self, train_set):
        """Fit the model to the training data, should be called before each implemention of any recommender model's class

        """

        self.train_set = train_set

        return self


    def score(self, user_id, item_id):
        """Predict the scores/ratings of a user for a list of items.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score predictions.
            
        item_id: int, required
            The index of the item for that to perform score predictions.

        Returns
        -------
        A scalar
            A relative score that the user gives to the item
        """

        raise NotImplementedError('The algorithm is not able to make score prediction!')


    def default_score(self):
        """Overwrite this function if your algorithm has special treatment for cold-start problem

        """

        return self.train_set.global_mean


    def default_rank(self, candidate_item_ids=None):
        """Overwrite this function if your algorithm has special treatment for cold-start problem

        """

        known_item_rank = self.train_set.item_ppl_rank

        if candidate_item_ids is None:
            rank_item_ids = known_item_rank
        else:
            known_candidate_items = intersects(known_item_rank, candidate_item_ids, assume_unique=True)
            unk_candidate_items = excepts(known_candidate_items, candidate_item_ids, assume_unique=True)
            rank_item_ids = np.concatenate((known_candidate_items, unk_candidate_items))

        return rank_item_ids


    def rate(self, user_id, item_id, clip=True):
        """Give a rating score between pair of user and item

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform item raking.

        item_id: int, required
            The index of the item to be rated by the user.

        Returns
        -------
        A scalar
            A rating score of the user for the item
        """

        try:
            rating_pred = self.score(user_id, item_id)
        except ScoreException:
            rating_pred = self.default_score()

        if clip:
            rating_pred = clipping(rating_pred, self.train_set.min_rating, self.train_set.max_rating)

        return rating_pred


    def rank(self, user_id, candidate_item_ids=None):
        """Rank all test items for a given user. To be re-implemented in each recommender model's class

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

        raise NotImplementedError('The algorithm is not able to rank items!')