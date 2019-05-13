# -*- coding: utf-8 -*-
"""
@author: Tran Thanh Binh
         
"""

from ..recommender import Recommender
from ...exception import ScoreException
import time
import math
import numpy as np


class CVAE(Recommender):
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

    def __init__(self, k=50, lamda_u=0.1, lamda_v=10, lamda_r=1, a=1, b=0.01, n_epochs=100, input_dim=8000,
        dimension=[200, 100], activations=['sigmoid', 'sigmoid'], n_z=50, loss_type='cross-entropy', lr=0.1,
        dropout=0.1, verbose=True, name="cvae", trainable=True, seed=None):

        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.lamda_u = lamda_u
        self.lamda_v = lamda_v
        self.lamda_r = lamda_r
        self.a = a
        self.b = b
        self.n_epochs = n_epochs
        self.input_dim = input_dim
        self.dimenssion = dimension
        self.n_z = n_z
        self.loss_type = loss_type
        self.activations = activations
        self.lr =lr
        self.k = k
        self.dropout = dropout
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

        rng = get_rng(self.seed)

        self.U = self.init_params.get('U', xavier_uniform((self.train_set.num_users, self.dimension), rng))
        self.V = self.init_params.get('V', xavier_uniform((self.train_set.num_items, self.dimension), rng))
        #self.W = self.init_params.get('W', xavier_uniform((self.train_set.item_text.vocab.size, self.emb_dim), rng))

        if self.trainable:
            self._fit_cvae()

    @staticmethod
    def _build_data(csr_mat):
        data = []
        index_list = []
        rating_list = []
        for i in range(csr_mat.shape[0]):
            j, k = csr_mat.indptr[i], csr_mat.indptr[i + 1]
            index_list.append(csr_mat.indices[j:k])
            rating_list.append(csr_mat.data[j:k])
        data.append(index_list)
        data.append(rating_list)
        return data

    def _fit_cvae(self):
        user_data = self._build_data(self.train_set.matrix)
        item_data = self._build_data(self.train_set.matrix.T.tocsr())

        n_user = len(user_data[0])
        n_item = len(item_data[0])

        # R_user and R_item contain rating values
        R_user = user_data[1]
        R_item = item_data[1]

        # Initialize cnn module
        from .convmf import CNN_module
        import tensorflow as tf
        from tqdm import trange

        cnn_module = CNN_module(output_dimension=self.dimension, dropout_rate=self.dropout_rate,
                                emb_dim=self.emb_dim, max_len=self.max_len,
                                nb_filters=self.num_kernel_per_ws, seed=self.seed, init_W=self.W)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())  # init variable

        document = self.train_set.item_text.batch_seq(np.arange(n_item), max_length=self.max_len)

        feed_dict = {cnn_module.model_input: document}
        theta = sess.run(cnn_module.model_output, feed_dict=feed_dict)

        endure = 3
        converge_threshold = 0.01
        history = 1e-50
        loss = 0

        for iter in range(self.n_epochs):
            print("Iteration {}".format(iter + 1))

            user_loss = np.zeros(n_user)
            VV = self.b * (self.V.T.dot(self.V)) + self.lamda_u * np.eye(self.k)

            for i in range(n_user):
                idx_item = user_data[0][i]
                V_i = self.V[idx_item]
                R_i = R_user[i]
                A = VV + (self.a-self.b)*(V_i.T.dot(V_i))
                x = (self.a * V_i * (np.tile(R_i, (self.k, 1)).T)).sum(0)
                self.U[i] = np.linalg.solve(A, x)

                user_loss[i] = -0.5 * self.lambda_u * np.dot(self.U[i], self.U[i])

            item_loss = np.zeros(n_item)
            UU = self.b * (self.U.T.dot(self.U))

            for j in range(n_item):
                idx_user = item_data[0][j]
                U_j = self.U[idx_user]
                R_j = R_item[j]

                tmp_A = UU + (self.a - self.b) * (U_j.T.dot(U_j))
                A = tmp_A + self.lambda_v * np.eye(self.k)
                x = (self.a * U_j * (np.tile(R_j, (self.k, 1)).T)).sum(0) + self.lambda_v * theta[j]
                self.V[j] = np.linalg.solve(A, x)

                item_loss[j] = -0.5 * np.square(R_j * self.a).sum()
                item_loss[j] = item_loss[j] + self.a * np.sum((U_j.dot(self.V[j])) * R_j)
                item_loss[j] = item_loss[j] - 0.5 * np.dot(self.V[j].dot(tmp_A), self.V[j])

            
            loop = trange(self.cnn_epochs, desc='CNN', disable=not self.verbose)
            for _ in loop:
                for batch_ids in self.train_set.item_iter(batch_size=128, shuffle=True):
                    batch_seq = self.train_set.item_text.batch_seq(batch_ids, max_length=self.max_len)
                    feed_dict = {cnn_module.model_input: batch_seq,
                                 cnn_module.v: self.V[batch_ids],
                                 cnn_module.sample_weight: item_weight[batch_ids]}

                    sess.run([cnn_module.optimizer], feed_dict=feed_dict)

            feed_dict = {cnn_module.model_input: document, cnn_module.v: self.V, cnn_module.sample_weight: item_weight}
            theta, cnn_loss = sess.run([cnn_module.model_output, cnn_module.weighted_loss], feed_dict=feed_dict)

            loss = loss + np.sum(user_loss) + np.sum(item_loss) - 0.5 * self.lambda_v * cnn_loss

            converge = abs((loss - history) / history)

            print("Loss: %.5f, Converge: %.6f " % (loss, converge))
            history = loss
            if converge < converge_threshold:
                endure -= 1
                if endure == 0:
                    break

        tf.reset_default_graph()

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

