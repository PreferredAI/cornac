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

        self.perfomance = None

    # fit the model to the traning data, should be re-implemented in each recommender model's class
    def fit(self, X):
        print('called outside of a recommender model!!!')



    def score(self, user_index, item_indexes = None):
        """Predict the scores/ratings of a user for a list of items.

        Parameters
        ----------
        user_index: int, required
            The index of the user for whom to perform score predictions.
            
        item_indexes: 1d array, optional, default: None
            A list of item indexes for which to predict the rating score.\
            When "None", score prediction is performed for all test items of the given user. 

        Returns
        -------
        Numpy 1d array 
            Array containing the predicted values for the items of interest
        """
        
        print('called outside of a recommender model!')

        return None
    
    

    def rank(self, user_index):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_index: int, required
            The index of the user for whom to perform item raking.
        known_items: 1d array, optional, default: None
            A list of item indices already known by the user

        Returns
        -------
        Numpy 1d array 
            Array of item indices sorted (in decreasing order) relative to some user preference scores. 
        """  
        
        print('called outside of a recommender model!')

        return None
