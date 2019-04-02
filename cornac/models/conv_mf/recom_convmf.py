# -*- coding: utf-8 -*-
"""
@author: Tran Thanh Binh
         
"""

from ..recommender import Recommender
import numpy as np
from .convmf import convmf
from ...exception import ScoreException


class ConvMF(Recommender):
    """
    Parameters
    ----------
    k: int, optional, default: 50
        The dimension of the user and item latent factors.

    n_epochs: int, optional, default: 50
        Maximum number of epochs for training.

    lambda_u: float, optional, default: 1.0
        The regularization hyper-parameter for user latent factor.

    lambda_v: float, optional, default: 100.0
        The regularization hyper-parameter for item latent factor.

    emb_dim: int, optional, default: 200
        The embedding size of each word. One word corresponds with [1 x emb_dim] vector in the embedding space

    max_len: int, optional, default 300
        The maximum length of item's document

    num_kernel_per_ws: int, optional, default: 100
        The number of kernel filter in convolutional layer

    dropout_rate: float, optional, default: 0.2
        Dropout rate while training CNN

    give_item_weight: boolean, optional, default: True
        When True, each item will be weighted base on the number of user who have rated this item

    init_params: dict, optional, default: {'U':None, 'V':None, 'W': None}
        Initial U and V matrix and initial weight for embedding layer W

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).

    References
    ----------
    * Donghyun Kim1, Chanyoung Park1. ConvMF: Convolutional Matrix Factorization for Document Context-Aware Recommendation. \
    In :10th ACM Conference on Recommender Systems Pages 233-240
    """

    def __init__(self, give_item_weight=True,
                 n_epochs=50, lambda_u=1, lambda_v=100, k=50,
                 name="convmf", trainable=True,
                 verbose=False, dropout_rate=0.2, emb_dim=200,
                 max_len=300, num_kernel_per_ws=100, init_params=None):

        Recommender.__init__(self, name='CONVMF', trainable=trainable)

        self.give_item_weight = give_item_weight
        self.max_iter = n_epochs
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.dimension = k
        self.dropout_rate = dropout_rate
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.num_kernel_per_ws = num_kernel_per_ws
        self.name = name
        self.verbose = verbose
        self.init_params = {} if init_params is None else init_params

    def fit(self, train_set):
        """Fit the model.

        Parameters
        ----------
        train_set: :obj:`cornac.data.MultimodalTrainSet`
            Multimodal training set.

        """
        Recommender.fit(self, train_set)

        if not self.trainable:
            print('%s is trained already (trainable = False)' % (self.name))
            return

        if self.verbose:
            print('Learning...')

        res = convmf(max_iter=self.max_iter,
                     lambda_u=self.lambda_u, lambda_v=self.lambda_v,
                     dimension=self.dimension, init_params=self.init_params,
                     give_item_weight=self.give_item_weight,
                     emb_dim=self.emb_dim,
                     num_kernel_per_ws=self.num_kernel_per_ws,
                     vocab_size=train_set.item_text.vocab.size, train_set=train_set)

        self.U = np.asarray(res['U'])
        self.V = np.asarray(res['V'])

        if self.verbose:
            print('Learning completed')

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
