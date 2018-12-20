# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import numpy as np
from ..recommender import Recommender
from .hpf import *
from ...exception import ScoreException


# HierarchicalPoissonFactorization: Hpf
class HPF(Recommender):
    """Hierarchical Poisson Factorization.

    Parameters
    ----------
    k: int, optional, default: 5
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations.

    name: string, optional, default: 'HPF'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained (Theta and Beta are not None). 
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: {'G_s':None, 'G_r':None, 'L_s':None, 'L_r':None}
        List of initial parameters, e.g., init_params = {'G_s':G_s, 'G_r':G_r, 'L_s':L_s, 'L_r':L_r}, \
        where G_s and G_r are of type csc_matrix or np.array with the same shape as Theta, see below). \
        They represent respectively the "shape" and "rate" parameters of Gamma distribution over \
        Theta. Similarly, L_s, L_r are the shape and rate parameters of the Gamma over Beta.
      
    Theta: csc_matrix, shape (n_users,k)
        The expected user latent factors.

    Beta: csc_matrix, shape (n_items,k)
        The expected item latent factors.

    References
    ----------
    * Gopalan, Prem, Jake M. Hofman, and David M. Blei. Scalable Recommendation with \
    Hierarchical Poisson Factorization. In UAI, pp. 326-335. 2015.
    """

    def __init__(self, k=5, max_iter=100, name="HPF", trainable=True,
                 verbose=False, init_params={'G_s': None, 'G_r': None, 'L_s': None, 'L_r': None}):
        Recommender.__init__(self, name=name, trainable=trainable, verbose = verbose)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter

        self.ll = np.full(max_iter, 0)
        self.etp_r = np.full(max_iter, 0)
        self.etp_c = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.Theta = None  # matrix of user factors
        self.Beta = None  # matrix of item factors


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

        if self.trainable:
            res = pf(X, k=self.k, max_iter=self.max_iter, init_param=self.init_params)
            self.Theta = np.asarray(res['Z'])
            self.Beta = np.asarray(res['W'])
        elif self.verbose:
            print('%s is trained already (trainable = False)' % (self.name))



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

        user_pred = self.Beta[item_id,:].dot(self.Theta[user_id, :])
        user_pred = np.array(user_pred, dtype='float64').flatten()[0]
        
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
            u_representation = np.ones(self.k)
        else:
            u_representation =  self.Theta[user_id, :]

        known_item_scores = self.Beta.dot(u_representation)
        known_item_scores = np.array(known_item_scores, dtype='float64').flatten()
        
        if candidate_item_ids is None:
            ranked_item_ids = known_item_scores.argsort()[::-1]
            return ranked_item_ids
        else:
            n_items = max(self.train_set.num_items, max(candidate_item_ids) + 1)
            user_pref_scores = np.ones(n_items) * np.sum(u_representation)
            user_pref_scores[:self.train_set.num_items] = known_item_scores

            ranked_item_ids = user_pref_scores.argsort()[::-1]
            mask = np.in1d(ranked_item_ids, candidate_item_ids)
            ranked_item_ids = ranked_item_ids[mask]

            return ranked_item_ids







