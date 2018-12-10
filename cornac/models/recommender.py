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

    """get prefiction for a single user (predictions for one user at a time for efficiency purposes)
       predictions are not stored for the same efficiency reasons
       should be re-implemented in each recommender model's class"""

    def predict(self, index_user):
        print('called outside of a recommender model!!!')

        return 0
