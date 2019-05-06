# -*- coding: utf-8 -*-
"""
@author: Tran Thanh Binh
         
"""

from ..recommender import Recommender
from ...exception import ScoreException
import time
import math
import numpy as np


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
                 max_len=300, num_kernel_per_ws=100, init_params=None, seed=None):

        super().__init__(name=name, trainable=trainable, verbose=verbose)

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
        self.seed = seed

    def fit(self, train_set):
        """Fit the model.

        Parameters
        ----------
        train_set: :obj:`cornac.data.MultimodalTrainSet`
            Multimodal training set.

        """
        Recommender.fit(self, train_set)

        from ...utils import get_rng
        from ...utils.init_utils import xavier_uniform


        self.seed = get_rng(self.seed)

        self.U = self.init_params.get('U', xavier_uniform((self.train_set.num_users, self.dimension), self.seed))
        self.V = self.init_params.get('V', xavier_uniform((self.train_set.num_items, self.dimension), self.seed))
        self.W = self.init_params.get('W', xavier_uniform((self.train_set.item_text.vocab.size, self.emb_dim), self.seed))

        if self.trainable:
            self._fit_convmf()

    def _fit_convmf(self,):

        from .convmf import CNN_module

        endure = 3
        converge_threshold = 0.01
        history = 1e-50
        loss = 0

        R_user = self.train_set.matrix
        user = []

        user_index_list = []
        user_rating_list = []
        for i in range(R_user.shape[0]):
            item_idx = R_user[i].nonzero()[1]
            rating = R_user[i, item_idx].A[0]
            user_index_list.append(item_idx)
            user_rating_list.append(rating)

        user.append(user_index_list)
        user.append(user_rating_list)

        R_item = self.train_set.matrix.tocsc().T
        item = []

        item_index_list = []
        item_rating_list = []
        for i in range(R_item.shape[0]):
            user_idx = R_item[i].nonzero()[1]
            rating = R_item[i, user_idx].A[0]
            item_index_list.append(user_idx)
            item_rating_list.append(rating)

        item.append(item_index_list)
        item.append(item_rating_list)

        n_user = len(user[0])
        n_item = len(item[0])

        # R_user and R_item contain rating values
        R_user = user[1]
        R_item = item[1]

        if self.give_item_weight:
            item_weight = np.array([math.sqrt(len(i))
                                    for i in R_item], dtype=float)
            item_weight = (float(n_item) / item_weight.sum()) * item_weight
        else:
            item_weight = np.ones(n_item, dtype=float)

        # Initialize cnn module
        cnn_module = CNN_module(output_dimesion=self.dimension,dropout_rate=self.dropout_rate,
                                emb_dim=self.emb_dim, max_len=self.max_len,
                                nb_filters=self.num_kernel_per_ws,seed=self.seed, init_W=self.W)

        document = self.train_set.item_text.batch_seq(np.arange(n_item), max_length=self.max_len)
        theta = cnn_module.get_projection_layer(document)

        for iter in range(self.max_iter):
            print("Iteration {}".format(iter + 1))
            tic = time.time()

            user_loss = np.zeros(n_user)
            for i in range(n_user):
                idx_item = user[0][i]
                V_i = self.V[idx_item]
                R_i = R_user[i]

                A = self.lambda_u * np.eye(self.dimension) + self._square(V_i)
                B = (V_i * (np.tile(R_i, (self.dimension, 1)).T)).sum(0)
                self.U[i] = np.linalg.solve(A, B)

                user_loss[i] = -0.5 * self.lambda_u * np.dot(self.U[i], self.U[i])

            item_loss = np.zeros(n_item)
            for j in range(n_item):
                idx_user = item[0][j]
                U_j = self.U[idx_user]
                R_j = R_item[j]
                Uj_square = self._square(U_j)

                A = self.lambda_v * item_weight[j] * np.eye(self.dimension) + Uj_square
                B = (U_j * (np.tile(R_j,(self.dimension, 1)).T)).sum(0) \
                    + self.lambda_v * item_weight[j] * theta[j]
                self.V[j] = np.linalg.solve(A, B)
                item_loss[j] = -0.5 * np.square(R_j).sum()
                item_loss[j] = item_loss[j] + np.sum((U_j.dot(self.V[j])) * R_j)
                item_loss[j] = item_loss[j] - 0.5 * np.dot(self.V[j].dot(Uj_square), self.V[j])

            cnn_loss = cnn_module.train(train_set=self.train_set, V=self.V, item_weight=item_weight)
            theta = cnn_module.get_projection_layer(X_train=document)
            loss = loss + np.sum(user_loss) + np.sum(item_loss) - 0.5 * self.lambda_v * cnn_loss * n_item
            toc = time.time()
            elapsed = toc - tic
            converge = abs((loss - history) / history)
            print("Loss: %.5f Elpased: %.4fs Converge: %.6f " % (loss, elapsed, converge))
            history = loss
            if converge < converge_threshold:
                endure -= 1
                if endure == 0:
                    break

    def _square(self, mat):
        """
        return XT.X matrix
        """
        return mat.T.dot(mat)

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


