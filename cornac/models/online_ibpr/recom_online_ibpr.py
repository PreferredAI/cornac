# -*- coding: utf-8 -*-
"""
@author: Dung D. Le (Andrew) <ddle.2015@smu.edu.sg>
"""

import numpy as np
from .online_ibpr import *
from ..recommender import Recommender


class OnlineIBPR(Recommender):
    """Online Indexable Bayesian Personalized Ranking.

    Parameters
    ----------
    k: int, optional, default: 20
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.05
        The learning rate for SGD.

    lamda: float, optional, default: 0.001
        The regularization parameter.

    batch_size: int, optional, default: 100
        The batch size for SGD.

    name: string, optional, default: 'IBRP'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V} \
        please see below the definition of U and V.

    U: csc_matrix, shape (n_users,k)
        The user latent factors, optional initialization via init_params.

    V: csc_matrix, shape (n_items,k)
        The item latent factors, optional initialization via init_params.

    References
    ----------
    * Le, D. D., & Lauw, H. W. (2017, November). Indexable Bayesian personalized ranking for efficient top-k recommendation.\
      In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (pp. 1389-1398). ACM.
    """

    def __init__(self, k=20, max_iter=100, learning_rate = 0.05, lamda = 0.001, batch_size = 100, name="online_ibpr", trainable = True, 
                 verbose=False, init_params = None):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter
        self.name = name
        self.learning_rate = learning_rate
        self.lamda = lamda   
        self.batch_size = batch_size 
        
        self.U = init_params['U']  # matrix of user factors
        self.V = init_params['V']  # matrix of item factors



    # fit the recommender model to the traning data
    def fit(self, train_set):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: object of type TrainSet, required
            An object contraining the user-item preference in csr scipy sparse format,\
            as well as some useful attributes such as mappings to the original user/item ids.\
            Please refer to the class TrainSet in the "data" module for details.
        """
        
        Recommender.fit(self, train_set)

        X = self.train_set.matrix        
        #change the data to original user Id item Id and rating format
        X = X.tocoo() # convert sparse matrix to COOrdiante format
        triplets = np.ndarray(shape=(len(X.data), 3), dtype=float)
        triplets[:, 0] = X.row
        triplets[:, 1] = X.col
        triplets[:, 2] = X.data
        
        if self.verbose:
            print('Learning...')
        res = online_ibpr(triplets, k=self.k, n_epochs=self.max_iter,lamda = self.lamda, learning_rate= self.learning_rate, batch_size = self.batch_size, init_params=self.init_params)
        self.U = np.asarray(res['U'])
        self.V = np.asarray(res['V'])
        if self.verbose:
            print('Learning completed')



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
            user_pred = self.U[user_index, :].dot(self.V[item_indexes,:].T)
        # transform user_pred to a flatten array
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