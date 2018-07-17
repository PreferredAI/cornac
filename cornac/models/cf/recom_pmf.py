# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np
import scipy.sparse as sp
import pmf
from ..recommender import Recommender


class Pmf(Recommender):
    """Probabilistic Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 5
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    lamda: float, optional, default: 0.01
        The regularization parameter.

    name: string, optional, default: 'PMF'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V} \
        please see below the definition of U and V.

    U: csc_matrix, shape (n_users,k)
        The user latent factors, optional initialization via init_params.

    V: csc_matrix, shape (n_items,k)
        The item latent factors, optional initialization via init_params.

    References
    ----------
    * Mnih, Andriy, and Ruslan R. Salakhutdinov. Probabilistic matrix factorization. \
    In NIPS, pp. 1257-1264. 2008.
    """

    def __init__(self, k=5, max_iter=100, learning_rate = 0.001, lamda = 0.01,name = "pmf",trainable = True,init_params = None):
        Recommender.__init__(self,name=name, trainable = trainable)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lamda = lamda
        
        self.ll = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.U = init_params['U'] #matrix of user factors
        self.V = init_params['V'] #matrix of item factors
        
        
    #fit the recommender model to the traning data    
    def fit(self,X):  
        
        if self.trainable:
            #converting data to the triplet format (needed for cython function pmf)
            (rid,cid,val)=sp.find(X)
            val = np.array(val,dtype='float32')
            rid = np.array(rid,dtype='int32')
            cid = np.array(cid,dtype='int32')
            tX = np.concatenate((np.concatenate(([rid], [cid]), axis=0).T,val.reshape((len(val),1))),axis = 1)
            del rid, cid, val
            print('Learning...')
            res = pmf.pmf(tX,k = self.k,n_X= X.shape[0], d_X =  X.shape[1], n_epochs = self.max_iter,lamda = self.lamda, learning_rate= self.learning_rate, init_params = self.init_params)
            self.U = sp.csc_matrix(res['U'])
            self.V = sp.csc_matrix(res['V'])
            print('Learning completed')
        else:
            print("The model is trained already (trainable = False)")
        
   
    

    #get prefiction for a single user (predictions for one user at a time for efficiency purposes)
    #predictions are not stored for the same efficiency reasons        
    def predict(self,index_user):
        user_pred = self.V.todense()*self.U[index_user,:].T.todense()
        #transform user_pred to a flatten array, but keep thinking about another possible format
        user_pred = np.array(user_pred,dtype='float64').flatten()
        
        return user_pred