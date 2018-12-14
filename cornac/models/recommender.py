# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:18:14 2017
@author: Aghiles Salah
"""


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
        print('called outside of a recommender model!!!')

        self.train_set = train_set


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
        
        print('called outside of a recommender model!')

        pass



    def rate(self, user_id, item_id, clip=True):
        pass


    def rank(self, user_id, candidate_item_ids):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform item raking.
        candidate_item_ids: 1d array, required
            A list of item indices to be ranked by the user

        Returns
        -------
        Numpy 1d array 
            Array of item indices sorted (in decreasing order) relative to some user preference scores. 
        """  
        
        print('called outside of a recommender model!')

        pass
