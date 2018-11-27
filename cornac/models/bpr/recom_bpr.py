# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao
"""

import numpy as np
from .bpr import *
from ..recommender import Recommender


class Bpr(Recommender):
    """Bayesian Personalized Ranking.

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

    batch_size: int, optional, default: 100
        The batch size for SGD.

    name: string, optional, default: 'BRP'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V} \
        please see below the definition of U and V.

    U: ndarray, shape (n_users,k)
        The user latent factors, optional initialization via init_params.

    V: ndarray, shape (n_items,k)
        The item latent factors, optional initialization via init_params.

    References
    ----------
    * Rendle, Steffen, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. \
    BPR: Bayesian personalized ranking from implicit feedback. In UAI, pp. 452-461. 2009.
    """

    def __init__(self, k=5, max_iter=100, learning_rate=0.001, lamda=0.01, batch_size=100, name="bpr", trainable=True, init_params=None):
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
        if self.trainable:
            #change the data to original user Id item Id and rating format
            cooX = X.tocoo()
            data = np.ndarray(shape=(len(cooX.data), 3), dtype=float)
            data[:, 0] = cooX.row
            data[:, 1] = cooX.col
            data[:, 2] = cooX.data

            print('Learning...')
            res = bpr(X, data, k=self.k, n_epochs=self.max_iter, lamda = self.lamda, learning_rate= self.learning_rate, batch_size = self.batch_size, init_params=self.init_params)
            self.U = res['U']
            self.V = res['V']
            print('Learning completed')
        else:
            print('%s is trained already (trainable = False)' % (self.name))

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
        
        user_pred = self.U[index_user, :].dot(self.V.T)
        # transform user_pred to a flatten array, but keep thinking about another possible format
        user_pred = np.array(user_pred, dtype='float64').flatten()

        return user_pred