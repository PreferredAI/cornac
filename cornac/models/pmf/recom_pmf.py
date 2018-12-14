# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np
import scipy.sparse as sp
import pmf
from ..recommender import Recommender
from ...utils.util_functions import sigmoid
from ...utils.util_functions import which_
from ...utils.util_functions import map_to
from ...utils.util_functions import clipping


class PMF(Recommender):
    """Probabilistic Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 5
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD_RMSProp.
        
    gamma: float, optional, default: 0.9
        The weight for previous/current gradient in RMSProp.

    lamda: float, optional, default: 0.001
        The regularization parameter.

    name: string, optional, default: 'PMF'
        The name of the recommender model.
        
    variant: {"linear","non_linear"}, optional, default: 'non_linear'
        Pmf variant. If 'non_linear', the Gaussian mean is the output of a Sigmoid function.\
        If 'linear' the Gaussian mean is the output of the identity function.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).
        
    rating_range: 1d array, optional, default: [None,None]
        The minimum and maximum rating values, e.g., [1,5]. 

    init_params: dictionary, optional, default: {'U':None,'V':None}
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}. \
        U: a csc_matrix of shape (n_users,k), containing the user latent factors. \
        V: a csc_matrix of shape (n_items,k), containing the item latent factors.

    References
    ----------
    * Mnih, Andriy, and Ruslan R. Salakhutdinov. Probabilistic matrix factorization. \
    In NIPS, pp. 1257-1264. 2008.
    """

    def __init__(self, k=5, max_iter=100, learning_rate = 0.001,gamma = 0.9, lamda = 0.001, name = "pmf", variant ='non_linear', trainable = True, rating_range = [None,None] ,init_params = {'U':None,'V':None}):
        Recommender.__init__(self,name=name, trainable = trainable)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lamda = lamda
        self.variant = variant
        
        self.ll = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.U = init_params['U'] #matrix of user factors
        self.V = init_params['V'] #matrix of item factors
        self.min_rating = rating_range[0]
        self.max_rating = rating_range[1]


    def fit_(self, train_set):
        self.train_set = train_set
        self.fit(train_set.matrix)

        
    #fit the recommender model to the traning data    
    def fit(self, X):
        """Fit the model to observations.

        Parameters
        ----------
        X: scipy sparse matrix, required
            the user-item preference matrix (traning data), in a scipy sparse format\
            (e.g., csc_matrix).
        """

        if self.min_rating is None:
            self.min_rating = np.min(X.data)
        if self.max_rating is None:
            self.max_rating = np.max(X.data)
        if self.trainable:
            #converting data to the triplet format (needed for cython function pmf)
            (rid,cid,val)=sp.find(X)
            val = np.array(val,dtype='float32')
            if self.variant == 'non_linear':   #need to map the ratings to [0,1]
                if[self.min_rating,self.max_rating] != [0,1]:
                    val = map_to(val,0.,1.,self.min_rating,self.max_rating)
            rid = np.array(rid,dtype='int32')
            cid = np.array(cid,dtype='int32')
            tX = np.concatenate((np.concatenate(([rid], [cid]), axis=0).T,val.reshape((len(val),1))),axis = 1)
            del rid, cid, val
            print('Learning...')
            if self.variant == 'linear':
                res = pmf.pmf_linear(tX,k = self.k,n_X= X.shape[0], d_X =  X.shape[1], n_epochs = self.max_iter,lamda = self.lamda, learning_rate= self.learning_rate,gamma = self.gamma, init_params = self.init_params)
            elif self.variant == 'non_linear':
                res = pmf.pmf_non_linear(tX,k = self.k,n_X= X.shape[0], d_X =  X.shape[1], n_epochs = self.max_iter,lamda = self.lamda, learning_rate= self.learning_rate,gamma = self.gamma, init_params = self.init_params)
            else:
                raise ValueError('variant must be one of {"linear","non_linear"}')
            self.U = sp.csc_matrix(res['U'])
            self.V = sp.csc_matrix(res['V'])
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
            user_pred = self.V.todense()*self.U[user_index,:].T.todense()
        else:
            user_pred = self.V[item_indexes,:].todense()*self.U[user_index,:].T.todense()
            
        user_pred = np.array(user_pred,dtype='float64').flatten()
        
        if self.variant == "non_linear":
            user_pred = sigmoid(user_pred)
            user_pred = map_to(user_pred,self.min_rating,self.max_rating,0.,1.)
        else:
            #perform clipping to enforce the predictions to lie in the same range as the original ratings
            user_pred = clipping(user_pred,self.min_rating,self.max_rating)        
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



    def score_(self, user_id, item_id):
        if not self.train_set.is_known_user(user_id) or not self.train_set.is_known_item(item_id):
            return self.train_set.global_mean

        mapped_uid = self.train_set.uid_map[user_id]
        mapped_iid = self.train_set.iid_map[item_id]
        user_pred = self.V[mapped_iid, :].todense() * self.U[mapped_uid, :].T.todense()

        if self.variant == "non_linear":
            user_pred = sigmoid(user_pred)
            user_pred = map_to(user_pred, self.min_rating, self.max_rating, 0., 1.)
        else:
            # perform clipping to enforce the predictions to lie in the same range as the original ratings
            user_pred = clipping(user_pred, self.min_rating, self.max_rating)

        return user_pred

    def rank_(self, user_id):
        if not self.train_set.is_known_user(user_id):
            return np.arange(self.train_set.num_items)

        user_pred = self.V.todense() * self.U[user_id, :].T.todense()
        user_pred = np.array(user_pred, dtype='float64').flatten()

        if self.variant == "non_linear":
            user_pred = sigmoid(user_pred)
            user_pred = map_to(user_pred, self.min_rating, self.max_rating, 0., 1.)
        else:
            # perform clipping to enforce the predictions to lie in the same range as the original ratings
            user_pred = clipping(user_pred, self.min_rating, self.max_rating)

        rank_item_list = (-user_pred).argsort()  # ordering the items (in decreasing order) according to the preference score

        return rank_item_list