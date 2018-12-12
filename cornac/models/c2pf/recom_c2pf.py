# -*- coding: utf-8 -*-
"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat, savemat
from ..recommender import Recommender
import c2pf


# Recommender class for Collaborative Context Poisson Factorization (C2PF)
class C2PF(Recommender):
    """Collaborative Context Poisson Factorization.

    Parameters
    ----------
    k: int, optional, default: 100
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations for variational C2PF.

    aux_info: array, required, shape (n_context_items,3)
        The item-context matrix, noted C in the original paper, \
        in the triplet sparse format: (row_id, col_id, value).

    variant: string, optional, default: 'c2pf'
        C2pf's variant: c2pf: 'c2pf', 'tc2pf' (tied-c2pf) or 'rc2pf' (reduced-c2pf). \
        Please refer to the original paper for details.

    name: string, optional, default: None
        The name of the recommender model. If None, \
        then "variant" is used as the default name of the model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (Theta, Beta and Xi are not None). 

    init_params: dictionary, optional, default: {'G_s':None, 'G_r':None, 'L_s':None, 'L_r':None, \
        'L2_s':None, 'L2_r':None, 'L3_s':None, 'L3_r':None}
        List of initial parameters, e.g., init_params = {'G_s':G_s, 'G_r':G_r, 'L_s':L_s, 'L_r':L_r, \
        'L2_s':L2_s, 'L2_r':L2_r, 'L3_s':L3_s, 'L3_r':L3_r}, \
        where G_s and G_r are of type csc_matrix or np.array with the same shape as Theta, see below). \
        They represent respectively the "shape" and "rate" parameters of Gamma distribution over \
        Theta. It is the same for L_s, L_r and Beta, L2_s, L2_r and Xi, L3_s, L3_r and Kappa.

    Theta: csc_matrix, shape (n_users,k)
        The expected user latent factors.

    Beta: csc_matrix, shape (n_items,k)
        The expected item latent factors.

    Xi: csc_matrix, shape (n_items,k)
        The expected context item latent factors multiplied by context effects Kappa, \
        please refer to the paper below for details.

    References
    ----------
    * Salah, Aghiles, and Hady W. Lauw. A Bayesian Latent Variable Model of User Preferences with Item Context. \
    In IJCAI, pp. 2667-2674. 2018.
    """

    def __init__(self, k=100, max_iter=100, aux_info=None, variant='c2pf', name=None, trainable=True,
                 init_params={'G_s': None, 'G_r': None, 'L_s': None, 'L_r': None, 'L2_s': None, 'L2_r': None,
                              'L3_s': None, 'L3_r': None}):
        if name is None:
            Recommender.__init__(self, name=variant.upper(), trainable=trainable)
        else:
            Recommender.__init__(self, name=name, trainable=trainable)

        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter

        self.ll = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.Theta = None  # user factors
        self.Beta = None  # item factors
        self.Xi = None  # context factors Xi multiplied by context effects Kappa
        self.aux_info = aux_info  # item-context matrix in the triplet sparse format: (row_id, col_id, value)
        self.variant = variant

    # fit the recommender model to the traning data
    def fit(self, X):
        """Fit the model to observations.

        Parameters
        ----------
        X: scipy sparse matrix, required
            the user-item preference matrix (traning data), in a scipy sparse format\
            (e.g., csc_matrix).
        """
        # recover the striplet sparse format from csc sparse matrix X (needed to feed c++)
        (rid, cid, val) = sp.find(X)
        val = np.array(val, dtype='float32')
        rid = np.array(rid, dtype='int32')
        cid = np.array(cid, dtype='int32')
        tX = np.concatenate((np.concatenate(([rid], [cid]), axis=0).T, val.reshape((len(val), 1))), axis=1)
        del rid, cid, val

        if self.variant == 'c2pf':
            res = c2pf.c2pf(tX, X.shape[0], X.shape[1], self.aux_info, X.shape[1], X.shape[1], self.k, self.max_iter,
                            self.init_params)
        elif self.variant == 'tc2pf':
            res = c2pf.t_c2pf(tX, X.shape[0], X.shape[1], self.aux_info, X.shape[1], X.shape[1], self.k, self.max_iter,
                              self.init_params)
        elif self.variant == 'rc2pf':
            res = c2pf.r_c2pf(tX, X.shape[0], X.shape[1], self.aux_info, X.shape[1], X.shape[1], self.k, self.max_iter,
                              self.init_params)
        else:
            res = c2pf.c2pf(tX, X.shape[0], X.shape[1], self.aux_info, X.shape[1], X.shape[1], self.k, self.max_iter,
                            self.init_params)

        self.Theta = sp.csc_matrix(res['Z']).todense()
        self.Beta = sp.csc_matrix(res['W']).todense()
        self.Xi = sp.csc_matrix(res['Q']).todense()


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

        if self.variant == 'c2pf' or self.variant == 'tc2pf':
            if item_indexes is None:
                user_pred = self.Beta * self.Theta[user_index, :].T + self.Xi * self.Theta[user_index, :].T
            else:
                user_pred = self.Beta[item_indexes,:] * self.Theta[user_index, :].T + self.Xi * self.Theta[user_index, :].T
        elif self.variant == 'rc2pf':
            if item_indexes is None:
                user_pred = self.Xi * self.Theta[user_index, :].T
            else:
                user_pred = self.Xi[item_indexes,] * self.Theta[user_index, :].T
        else:
            if item_indexes is None:
                user_pred = self.Beta * self.Theta[user_index, :].T + self.Xi * self.Theta[user_index, :].T
            else:
                user_pred = self.Beta[item_indexes,:] * self.Theta[user_index, :].T + self.Xi * self.Theta[user_index, :].T            
        # transform user_pred to a flatten array,
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
