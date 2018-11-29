# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao
"""

from ..recommender import Recommender
from .autorec import *
import numpy as np
from ...utils.util_functions import *


class Autorec(Recommender):
    """Autoencoders Meet Collaborative Filtering

    Parameters
    ----------
    k: int, optional, default: 20
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    lamda: float, optional, default: 0.01
        The regularization parameter.

    batch_size: int, optional, default: 100
        The batch size of users for training.

    name: string, optional, default: 'autorec'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (W and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V} \
        please see below the definition of U and V.

    V: ndarray, shape (k,n_items)
        The encoder transformation matrix, optional initialization via init_params.

    W: ndarray, shape (n_items,k)
        The decoder transformation matrix, optional initialization via init_params.

    E: ndarray, shape (k,n_users)
    The encoded ratings, optional initialization via init_params.

    mu: ndarray, shape (k,1)
    The encoder bias, optional initialization via init_params.

    b: ndarray, shape (n_items,1)
    The decoder bias, optional initialization via init_params.

    g_act: encoder active function

    f_act: decoder active function

    References
    ----------
    * S. Sedhain, A. K. Menon, S. Sanner, and L. Xie. \
    Autorec: Autoencoders meet collaborative filtering.
    In Proceedings of the 24th International Conference on World Wide Web, WWW ’15 Companion, pages 111–112, New York, NY, USA, 2015. ACM.
    """

    def __init__(self, k=20, max_iter=100, learning_rate=0.001, lamda=0.01, batch_size=50, name="autorec",
                 trainable=True, g_act="Sigmoid", f_act="Identity", init_params=None):
        Recommender.__init__(self, name=name, trainable=trainable)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter
        self.name = name
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.batch_size = batch_size

        self.V = init_params['V']    # encoder transformation matrix
        self.mu = init_params['mu']  # encoder bias
        self.W = init_params['W']    # decoder transformation matrix
        self.b = init_params['b']    # decoder bias
        self.E = init_params['E']    # encoded ratings

        self.g_act = g_act
        self.f_act = f_act

    # fit the recommender model to the traning data
    def fit(self, X):
        # change the data to original user Id item Id and rating format
        data = np.ndarray(shape=(len(X.data), 3), dtype=float)
        data[:, 0] = X.tocoo().row
        data[:, 1] = X.tocoo().col
        data[:, 2] = X.data

        print('Learning...')
        res = autorec(X, data, k=self.k, n_epochs=self.max_iter, lamda=self.lamda, learning_rate=self.learning_rate,
                  batch_size=self.batch_size, g_act=self.g_act, f_act=self.f_act, init_params=self.init_params)
        self.V = res['V']
        self.W = res['W']
        self.mu = res['mu']
        self.b = res['b']
        self.E = res['E']

        print('Learning completed')

    # get prefiction for a single user (predictions for one user at a time for efficiency purposes)
    # predictions are not stored for the same efficiency reasons"""

    def predict(self, index_user):

        def sigmoid(x):
            return 1 / (1 + np.exp(x))

        def idetity(x):
            return x

        def relu(x):
            return np.maximum(x, 0)

        if self.g_act == "Sigmoid":
            g_act = sigmoid
        elif self.g_act == "Relu":
            g_act = relu
        elif self.g_act == "Tanh":
            g_act = np.tanh
        elif self.g_act == "Identity":
            g_act = idetity
        else:
            raise NotImplementedError("Active function ERROR")

        if self.f_act == "Sigmoid":
            f_act = sigmoid
        elif self.f_act == "Relu":
            f_act = relu
        elif self.f_act == "Tanh":
            f_act = np.tanh
        elif self.f_act == "Identity":
            f_act = idetity
        else:
            raise NotImplementedError("Active function ERROR")

        user_pred = f_act(self.W.dot(g_act(self.E[:, index_user] + self.mu.flatten()))+self.b.flatten())

        # transform user_pred to a flatten array, but keep thinking about another possible format
        user_pred = np.array(user_pred, dtype='float64').flatten()
        user_pred = clipping(user_pred, 1, 5)

        return user_pred