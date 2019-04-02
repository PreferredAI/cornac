# -*- coding: utf-8 -*-

"""
@author: Trieu Thi Ly Ly 
"""

from ..recommender import Recommender
from .cdl import *
from ...exception import ScoreException


class CDL(Recommender):
    """Collaborative Deep Learning.

    Parameters
    ----------
    k: int, optional, default: 50
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    autoencoder_structure：array, optional, default: [200]
        The number of neurons of encoder/ decoder layer for SDAE

    learning_rate: float, optional, default: 0.001
        The learning rate for AdamOptimizer.

    lambda_u: float, optional, default: 0.1
        The regularization parameter for users.

    lambda_v: float, optional, default: 10
        The regularization parameter for items.

    lambda_w: float, optional, default: 0.1
        The regularization parameter for SDAE weights.

    lambda_n: float, optional, default: 1000
        The regularization parameter for SDAE output.

    a: float, optional, default: 1
        The confidence of observed ratings.

    b: float, optional, default: 0.01
        The confidence of unseen ratings.

    autoencoder_corruption: float, optional, default: 0.3
        The corruption ratio for SDAE.

    dropout_rate: float, optional, default: 0.1
        The probability that each element is removed in dropout of SDAE.

    batch_size: int, optional, default: 100
        The batch size for SGD.

    name: string, optional, default: 'CDL'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already 
        pre-trained (U and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}
        U: ndarray, shape (n_users,k)
            The user latent factors, optional initialization via init_params.
        V: ndarray, shape (n_items,k)
            The item latent factors, optional initialization via init_params.

    References
    ----------
    * Hao Wang, Naiyan Wang, Dit-Yan Yeung. CDL: Collaborative Deep Learning for Recommender Systems. In : SIGKDD. 2015. p. 1235-1244.
    """

    def __init__(self, k=50, autoencoder_structure=None, lambda_u=0.1, lambda_v=100,
                 lambda_w=0.1, lambda_n=1000, a=1, b=0.01, autoencoder_corruption=0.3, learning_rate=0.001,
                 dropout_rate=0.1, batch_size=128, max_iter=100, name="CDL", trainable=True, verbose=True,
                 init_params=None):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w
        self.lambda_n = lambda_n
        self.a = a
        self.b = b
        self.autoencoder_corruption = autoencoder_corruption
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.name = name
        self.init_params = init_params
        self.max_iter = max_iter
        self.autoencoder_structure = autoencoder_structure
        self.batch_size = batch_size
        self.verbose = verbose

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
        if not self.trainable:
            print('%s is trained already (trainable = False)' % (self.name))
            return

        if self.verbose:
            print('Learning...')

        self.U, self.V = cdl(train_set, self.autoencoder_structure, k=self.k, lambda_u=self.lambda_u,
                             lambda_v=self.lambda_v, lambda_w=self.lambda_w, lambda_n=self.lambda_n,
                             a=self.a, b=self.b, corruption_rate=self.autoencoder_corruption,
                             n_epochs=self.max_iter, lr=self.learning_rate, dropout_rate=self.dropout_rate,
                             batch_size=self.batch_size, init_params=self.init_params, verbose=self.verbose)

        if self.verbose:
            print('\nLearning completed')

    def score(self, user_id, item_id=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_id: int, required
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
            if self.train_set.is_unk_user(user_id):
                raise ScoreException("Can't make score prediction for (user_id=%d)" % user_id)

            known_item_scores = self.V.dot(self.U[user_id, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
            user_pred = self.V[item_id, :].dot(self.U[user_id, :])

            return user_pred
