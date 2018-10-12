# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import numpy as np
from ..recommender import Recommender
from .hpf import *

#HierarchicalPoissonFactorization: Hpf
class Hpf(Recommender):
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
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (Theta and Beta are not None). 

    init_params: dictionary, optional, default: None
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

    def __init__(self, k=5, max_iter=100,name = "HPF",trainable = True, init_params = None):
        Recommender.__init__(self,name=name, trainable = trainable)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter
        
        self.ll = np.full(max_iter, 0)
        self.etp_r = np.full(max_iter, 0)
        self.etp_c = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.Theta = None #matrix of user factors
        self.Beta = None #matrix of item factors
        
        
    #fit the recommender model to the traning data    
    def fit(self,X):
        """Fit the model to observations.

        Parameters
        ----------
        X: scipy sparse matrix, required
            the user-item preference matrix (traning data), in a scipy sparse format\
            (e.g., csc_matrix).
        """
        if self.trainable: 
            res = pf(X,k = self.k, max_iter = self.max_iter,init_param = self.init_params)
            self.Theta = res['Z']
            self.Beta = res['W']
        else:
            print('%s is trained already (trainable = False)' % (self.name))
        
   
    

    #get prefiction for a single user (predictions for one user at a time for efficiency purposes)
    #predictions are not stored for the same efficiency reasons"""         
    def predict(self,index_user):
        """Predic the scores (ratings) of a user for all items.

        Parameters
        ----------
        index_user: int, required
            The index of the user for whom to perform predictions.

        Returns
        -------
        Numpy 1d array 
            Array containing the predicted values for all items
        """
        
        user_pred = self.Beta*self.Theta[index_user,:].T
        #transform user_pred to a flatten array, but keep thinking about another possible format
        user_pred = np.array(user_pred,dtype='float64').flatten()
        return user_pred
                
        
        