# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:18:14 2017
@author:    Aghiles Salah
            Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..exception import ScoreException


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

    def __init__(self, name, trainable=True):
        self.name = name
        self.trainable = trainable

        self.train_set = None
        self.perfomance = None

    # fit the model to the traning data, should be re-implemented in each recommender model's class
    def fit(self, train_set):

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



    def rate(self, user_id, item_id, clip=True):
        """

        """

        try:
            pred_rating = self.score(user_id, item_id)
        except ScoreException:
            pred_rating = self.default_score()

        if clip:
            pred_rating = max(pred_rating, self.train_set.min_rating)
            pred_rating = min(pred_rating, self.train_set.max_rating)

        return pred_rating


    def rank(self, user_id, candidate_item_ids=None):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform item raking.

        candidate_item_ids: 1d array, optional, default: None
            A list of item indices to be ranked by the user.
            If None, list of ranked known item indices will be returned

        Returns
        -------
        Numpy 1d array 
            Array of item indices sorted (in decreasing order) relative to some user preference scores.
        """

        raise NotImplementedError('The algorithm is not able to rank items!')