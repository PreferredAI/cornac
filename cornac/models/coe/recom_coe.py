# -*- coding: utf-8 -*-
"""
@author: Dung D. Le (Andrew) <ddle.2015@smu.edu.sg>
"""

import numpy as np
from  .coe import *
from ..recommender import Recommender


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

    def __init__(self, k=20, max_iter=100, learning_rate = 0.05, lamda = 0.001, batch_size = 1000, name="coe",trainable = True,init_params = None):
        Recommender.__init__(self, name=name, trainable = trainable)
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
    def fit(self, X):
        """Fit the model to observations.

        Parameters
        ----------
        X: scipy sparse matrix, required
            the user-item preference matrix (traning data), in a scipy sparse format\
            (e.g., csc_matrix).
        """
        #change the data to original user Id item Id and rating format
        #X = X.tocoo() # convert sparse matrix to COOrdiante format
        #data = np.ndarray(shape=(len(X.data), 3), dtype=float)
        #data[:, 0] = X.row
        #data[:, 1] = X.col
        #data[:, 2] = X.data

        print('Learning...')
        res = coe(X, k=self.k, n_epochs=self.max_iter,lamda = self.lamda, learning_rate= self.learning_rate, batch_size = self.batch_size, init_params=self.init_params)
        self.U = res['U']
        self.V = res['V']
        print('Learning completed')

    #get prefiction for a single user (predictions for one user at a time for efficiency purposes)
    #predictions are not stored for the same efficiency reasons"""

    def predict(self, index_user):
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
        user_pred = np.sum(np.abs(self.V - self.U[index_user, :])**2,axis=-1)**(1./2) 
        # transform user_pred to a flatten array, but keep thinking about another possible format
        user_pred = np.array(user_pred, dtype='float64').flatten()

        return user_pred