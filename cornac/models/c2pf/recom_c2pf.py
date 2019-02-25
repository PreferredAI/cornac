# -*- coding: utf-8 -*-
"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import numpy as np
import scipy.sparse as sp
from ..recommender import Recommender
import c2pf
from ...exception import ScoreException



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

    def __init__(self, k=100, max_iter=100, aux_info=None, variant='c2pf', name=None, trainable=True, verbose=False,
                 init_params={'G_s': None, 'G_r': None, 'L_s': None, 'L_r': None, 'L2_s': None, 'L2_r': None,
                              'L3_s': None, 'L3_r': None}):
        if name is None:
            Recommender.__init__(self, name=variant.upper(), trainable=trainable, verbose=verbose)
        else:
            Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)

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

    #fit the recommender model to the traning data    
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
        X = sp.csc_matrix(self.train_set.matrix)
        
        # recover the striplet sparse format from csc sparse matrix X (needed to feed c++)
        (rid, cid, val) = sp.find(X)
        val = np.array(val, dtype='float32')
        rid = np.array(rid, dtype='int32')
        cid = np.array(cid, dtype='int32')
        tX = np.concatenate((np.concatenate(([rid], [cid]), axis=0).T, val.reshape((len(val), 1))), axis=1)
        del rid, cid, val
        
        
        if self.trainable:
            # align auxiliary information with training data
            raw_iid = train_set.get_raw_iid_list()
            map_iid = train_set._iid_map
            c_iid = []
            c_cid = []
            c_val = []

            for i, j, _ in self.aux_info:
                if (not i in raw_iid) or (not j in raw_iid):
                    continue
                c_iid.append(map_iid[i])
                c_cid.append(map_iid[j])
                c_val.append(1.0)

            c_val = np.array(c_val,dtype='float32')
            c_iid = np.array(c_iid,dtype='int32')
            c_cid = np.array(c_cid,dtype='int32')

            train_aux_info = np.concatenate((np.concatenate(([c_iid], [c_cid]), axis=0).T,c_val.reshape((len(c_val),1))),axis = 1)
            del c_iid, c_cid, c_val

            
            
            if self.variant == 'c2pf':
                res = c2pf.c2pf(tX, X.shape[0], X.shape[1], train_aux_info, X.shape[1], X.shape[1], self.k, self.max_iter,
                            self.init_params)
            elif self.variant == 'tc2pf':
                res = c2pf.t_c2pf(tX, X.shape[0], X.shape[1], train_aux_info, X.shape[1], X.shape[1], self.k, self.max_iter,
                              self.init_params)
            elif self.variant == 'rc2pf':
                res = c2pf.r_c2pf(tX, X.shape[0], X.shape[1], train_aux_info, X.shape[1], X.shape[1], self.k, self.max_iter,
                              self.init_params)
            else:
                res = c2pf.c2pf(tX, X.shape[0], X.shape[1], train_aux_info, X.shape[1], X.shape[1], self.k, self.max_iter,
                            self.init_params)

            self.Theta = sp.csc_matrix(res['Z']).todense()
            self.Beta = sp.csc_matrix(res['W']).todense()
            self.Xi = sp.csc_matrix(res['Q']).todense()
        elif self.verbose:
            print('%s is trained already (trainable = False)' % (self.name))        
        
        

<<<<<<< HEAD
    def score(self, user_id, item_id):
        """Predict the scores/ratings of a user for a list of items.
=======
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
            
        if self.variant == 'c2pf' or self.variant == 'tc2pf':
            user_pred = self.Beta[item_id,:].dot(self.Theta[user_id,:]) + self.Xi[item_id,:].dot(self.Theta[user_id,:])
        elif self.variant == 'rc2pf':
            user_pred = self.Xi[item_id,:].dot(self.Theta[user_id,:])
        else:
            user_pred = self.Beta[item_id,:].dot(self.Theta[user_id, :]) + self.Xi[item_id,:].dot(self.Theta[user_id,:])            

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
            u_representation =  self.Theta[user_id, :].T
                        
        if self.variant == 'c2pf' or self.variant == 'tc2pf':
            known_item_scores = self.Beta.dot(u_representation) + self.Xi.dot(u_representation)
        elif self.variant == 'rc2pf':
            known_item_scores = self.Xi.dot(u_representation)
        else:
            known_item_scores = self.Beta.dot(u_representation) + self.Xi.dot(u_representation)

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
        if self.variant == 'c2pf' or self.variant == 'tc2pf':
            if item_id is None:
                user_pred = self.Beta * self.Theta[user_id, :].T + self.Xi * self.Theta[item_id, :].T
            else:
                user_pred = self.Beta[item_id,:] * self.Theta[user_id, :].T + self.Xi * self.Theta[user_id, :].T
        elif self.variant == 'rc2pf':
            if item_id is None:
                user_pred = self.Xi * self.Theta[user_id, :].T
            else:
                user_pred = self.Xi[item_id,] * self.Theta[user_id, :].T
        else:
            if item_id is None:
                user_pred = self.Beta * self.Theta[user_id, :].T + self.Xi * self.Theta[user_id, :].T
            else:
                user_pred = self.Beta[item_id,:] * self.Theta[user_id, :].T + self.Xi * self.Theta[user_id, :].T
        # transform user_pred to a flatten array,
        user_pred = np.array(user_pred, dtype='float64').flatten()

        return user_pred
>>>>>>> upstream/master
