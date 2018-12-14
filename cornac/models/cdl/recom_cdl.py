# -*- coding: utf-8 -*-

"""
@author: Trieu Thi Ly Ly 
"""

from ..recommender import Recommender
from .cdl import *

class CDL(Recommender):
    """Collaborative Deep Learning.

    Parameters
    ----------
    k: int, optional, default: 50
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    text_information：ndarray, shape (n_items, n_vocabularies), optional, default:None
        Bag-of-words features of items

    autoencoder_structure：array, optional, default: [200]
        The number of neurons of encoder/ decoder layer for SDAE

    learning_rate: float, optional, default: 0.001
        The learning rate for AdamOptimizer.

    lambda_u: float, optional, default: 0.1
        The regularization parameter for users.

    lambda_v: float, optional, default: 10
        The regularization parameter for items.

    lambda_w: float, optional, default: 0.1
        The regularization parameter for SDAE weights.

    lambda_n: float, optional, default: 1000
        The regularization parameter for SDAE output.

    a: float, optional, default: 1
        The confidence of observed ratings.

    b: float, optional, default: 0.01
        The confidence of unseen ratings.

    autoencoder_corruption: float, optional, default: 0.3
        The corruption ratio for SDAE.

    keep_prob: float, optional, default: 1.0
        The probability that each element is kept in dropout of SDAE.

    batch_size: int, optional, default: 100
        The batch size for SGD.

    name: string, optional, default: 'CDL'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already 
        pre-trained (U and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V} 
        please see below the definition of U and V.

    U: ndarray, shape (n_users,k)
        The user latent factors, optional initialization via init_params.

    V: ndarray, shape (n_items,k)
        The item latent factors, optional initialization via init_params.

    References
    ----------
    * Hao Wang, Naiyan Wang, Dit-Yan Yeung. CDL: Collaborative Deep Learning for Recommender Systems. In : SIGKDD. 2015. p. 1235-1244.
    """
    
    def __init__(self, k=50, text_information = None, autoencoder_structure = None ,lambda_u = 0.1, lambda_v = 0.01,lambda_w = 0.01, lambda_n = 0.01, a = 1, b = 0.01, autoencoder_corruption = 0.3, learning_rate=0.001, keep_prob = 1.0, batch_size = 100, max_iter=100, name = "CDL",trainable = True, init_params = None):
        Recommender.__init__(self,name=name, trainable = trainable)
        self.k = k
        self.text_information = text_information
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w
        self.lambda_n = lambda_n
        self.a = a
        self.b = b
        self.autoencoder_corruption = autoencoder_corruption
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.name = name
        self.init_params = init_params
        self.max_iter = max_iter
        self.autoencoder_structure =  autoencoder_structure
        self.batch_size = batch_size 

        self.U = init_params['U']  # matrix of user factors
        self.V = init_params['V']  # matrix of item factors
        
    #fit the recommender model to the traning data    
    def fit(self,X):
        """Fit the model to observations.

        Parameters
        ----------
        X: scipy sparse matrix, required
            the user-item preference matrix (training data), in a scipy sparse format\
            (e.g., csc_matrix).

        (e.g., csc_matrix).
        """
        if self.trainable:
            res = cdl(X, self.text_information, self.autoencoder_structure, k = self.k, lambda_u = self.lambda_u, lambda_v = self.lambda_v, lambda_w = self.lambda_w, lambda_n = self.lambda_n , a = self.a, b = self.b, autoencoder_corruption = self.autoencoder_corruption, n_epochs=self.max_iter, learning_rate= self.learning_rate, keep_prob = self.keep_prob, batch_size = self.batch_size, init_params = self.init_params)
            self.U = res['U']
            self.V = res['V']
            print('Learning completed')
        else:
            print('%s is trained already (trainable = False)' % (self.name))
        
                
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
        
        if item_indexes is None: 
            user_pred = self.U[user_index, :].dot(self.V.T)
        else:
            user_pred = self.U[user_index,:].dot(self.V[item_indexes,:].T)
        # transform user_pred to a flatten array, but keep thinking about another possible format
        user_pred = np.array(user_pred, dtype='float64').flatten()

        return user_pred



    def rank(self, user_index, known_items = None):
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
        
        u_pref_score = np.array(self.score(user_index))
        if known_items is not None:
            u_pref_score[known_items] = None
            
        rank_item_list = (-u_pref_score).argsort()  # ordering the items (in decreasing order) according to the preference score

        return rank_item_list