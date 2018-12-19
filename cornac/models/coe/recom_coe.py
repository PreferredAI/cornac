# -*- coding: utf-8 -*-
"""
@author: Dung D. Le (Andrew) <ddle.2015@smu.edu.sg>
"""

import numpy as np
from  .coe import *
from ..recommender import Recommender
from ...exception import ScoreException



class COE(Recommender):
    """Collaborative Ordinal Embedding.

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
    * Le, D. D., & Lauw, H. W. (2016, June). Euclidean co-embedding of ordinal data for multi-type visualization.\
     In Proceedings of the 2016 SIAM International Conference on Data Mining (pp. 396-404). Society for Industrial and Applied Mathematics.
    """



    def __init__(self, k=20, max_iter=100, learning_rate = 0.05, lamda = 0.001, batch_size = 1000, name="coe",trainable = True, 
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

        if self.verbose:
            print('Learning...')
        res = coe(X, k=self.k, n_epochs=self.max_iter,lamda = self.lamda, learning_rate= self.learning_rate, batch_size = self.batch_size, init_params=self.init_params)
        self.U = np.asarray(res['U'])
        self.V = np.asarray(res['V'])
        
        if self.verbose:
            print('Learning completed')

    #get prefiction for a single user (predictions for one user at a time for efficiency purposes)
    #predictions are not stored for the same efficiency reasons"""

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
        
        if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
            raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
        

        user_pred = np.sum(np.abs(self.V[item_id,:] - self.U[user_id, :])**2,axis=-1)**(1./2) 


        return user_pred
    
    
 
    
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
        
        if self.train_set.is_unk_user(user_id):
            if candidate_item_ids is None:
                return np.arange(self.train_set.num_items)
            return candidate_item_ids
        
        
        known_item_scores = np.sum(np.abs(self.V - self.U[user_id, :])**2,axis=-1)**(1./2)
        
        if candidate_item_ids is None:
            ranked_item_ids = known_item_scores.argsort()[::-1]
            return ranked_item_ids
        else:
            num_items = max(self.train_set.num_items, max(candidate_item_ids) + 1)
            user_pref_scores = np.ones(num_items) * self.default_score()
            user_pref_scores[:self.train_set.num_items] = known_item_scores

            ranked_item_ids = user_pref_scores.argsort()[::-1]
            mask = np.in1d(ranked_item_ids, candidate_item_ids)
            ranked_item_ids = ranked_item_ids[mask]

            return ranked_item_ids 