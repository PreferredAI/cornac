# -*- coding: utf-8 -*-
"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import numpy as np
from ..recommender import Recommender
from .pcrl import PCRL_
import scipy.sparse as sp
from ...exception import ScoreException




# Recommender class for Probabilistic Collaborative Representation Learning (PCRL)
class PCRL(Recommender):
    """Probabilistic Collaborative Representation Learning.

    Parameters
    ----------
    k: int, optional, default: 100
        The dimension of the latent factors.
        
    z_dims: Numpy 1d array, optional, default: [300]
        The dimensions of the hidden intermdiate layers 'z' in the order \
        [dim(z_L), ...,dim(z_1)], please refer to Figure 1 in the orginal paper for more details.

    max_iter: int, optional, default: 300
        Maximum number of iterations (number of epochs) for variational PCRL.
        
    batch_size: int, optional, default: 300
        The batch size for SGD.
        
    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    aux_info: csc sparse matrix, required
        The item auxiliary information matrix, item-context in the PCRL's paper, \
        in the scipy csc sparse format.

    name: string, optional, default: 'PCRL'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (Theta, Beta and Xi are not None). 
        
    w_determinist: boolean, optional, default: True
        When True, determinist wheights "W" are used for the generator network, \
        otherwise "W" is stochastic as in the original paper.

    init_params: dictionary, optional, default: {'G_s':None, 'G_r':None, 'L_s':None, 'L_r':None}
        List of initial parameters, e.g., init_params = {'G_s':G_s, 'G_r':G_r, 'L_s':L_s, 'L_r':L_r}, \
        where G_s and G_r are of type csc_matrix or np.array with the same shape as Theta, see below). \
        They represent respectively the "shape" and "rate" parameters of Gamma distribution over \
        Theta. It is the same for L_s, L_r and Beta.

    Theta: csc_matrix, shape (n_users,k)
        The expected user latent factors.

    Beta: csc_matrix, shape (n_items,k)
        The expected item latent factors.

    References
    ----------
    * Salah, Aghiles, and Hady W. Lauw. Probabilistic Collaborative Representation Learning for Personalized Item Recommendation. \
    In UAI 2018.
    """

<<<<<<< HEAD
    def __init__(self, k=100, z_dims = [300], max_iter=300, batch_size = 300,learning_rate = 0.001,aux_info = None, name = "pcrl", trainable = True,
                 verbose=False, w_determinist = True, init_params = {'G_s':None, 'G_r':None, 'L_s':None, 'L_r':None}):

        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
=======
    def __init__(self, k=100, z_dims=[300], max_iter=300, batch_size=300, learning_rate=0.001, aux_info=None,
                 name="pcrl", trainable=True, w_determinist=True,
                 init_params={'G_s': None, 'G_r': None, 'L_s': None, 'L_r': None}):

        Recommender.__init__(self, name=name, trainable=trainable)
>>>>>>> upstream/master

        self.aux_info = aux_info
        self.k = k
        self.z_dims = z_dims  # the dimension of the second hidden layer (we consider a 2-layers PCRL)
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_params = init_params
        self.w_determinist = w_determinist

<<<<<<< HEAD
        
        
    #fit the recommender model to the traning data    
    def fit(self, train_set):
=======
    # fit the recommender model to the traning data
    def fit(self, X):
>>>>>>> upstream/master
        """Fit the model to observations.

        Parameters
        ----------
        train_set: object of type TrainSet, required
            An object contraining the user-item preference in csr scipy sparse format,\
            as well as some useful attributes such as mappings to the original user/item ids.\
            Please refer to the class TrainSet in the "data" module for details.
        """

        Recommender.fit(self, train_set)
        X = sp.csc_matrix(self.train_set.matrix)
        
        if self.trainable:
            # intanciate pcrl
            pcrl_ = PCRL_(cf_data=X, aux_data=self.aux_info, k=self.k, z_dims=self.z_dims, n_epoch=self.max_iter,
                          batch_size=self.batch_size, learning_rate=self.learning_rate, B=1,
                          w_determinist=self.w_determinist, init_params=self.init_params)
            pcrl_.learn()
<<<<<<< HEAD
                        
            self.Theta = np.array(pcrl_.Gs)/np.array(pcrl_.Gr)
            self.Beta = np.array(pcrl_.Ls)/np.array(pcrl_.Lr)
        elif self.verbose:
            print('%s is trained already (trainable = False)' % (self.name))             

      
    def score(self, user_id, item_id):
        """Predict the scores/ratings of a user for a list of items.
=======

            self.Theta = np.array(pcrl_.Gs) / np.array(pcrl_.Gr)
            self.Beta = np.array(pcrl_.Ls) / np.array(pcrl_.Lr)
        else:
            print('%s is trained already (trainable = False)' % (self.name))

    def score(self, user_id, item_id=None):
        """Predict the scores/ratings of a user for an item.
>>>>>>> upstream/master

        Parameters
        ----------
        user_id: int, required
<<<<<<< HEAD
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
        
    
    def rank(self, user_id, candidate_item_ids = None):
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
    
    
    
    
    
    
    
=======
            The index of the user for whom to perform score prediction.

        item_id: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """

        if item_id is None:
            user_pred = self.Beta * self.Theta[user_id, :].T
        else:
            user_pred = self.Beta[item_id, :] * self.Theta[user_id, :].T
        # transform user_pred to a flatten array
        user_pred = np.array(user_pred, dtype='float64').flatten()

        return user_pred
>>>>>>> upstream/master
