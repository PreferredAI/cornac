# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import time
import math

import numpy as np

from ..recommender import Recommender
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

    def __init__(self, give_item_weight=True, cnn_epochs=5,
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
        self.cnn_epochs = cnn_epochs
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

        rng = get_rng(self.seed)

        self.U = self.init_params.get('U', xavier_uniform((self.train_set.num_users, self.dimension), rng))
        self.V = self.init_params.get('V', xavier_uniform((self.train_set.num_items, self.dimension), rng))
        self.W = self.init_params.get('W', xavier_uniform((self.train_set.item_text.vocab.size, self.emb_dim), rng))

        if self.trainable:
            self._fit_convmf()

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

    def _fit_convmf(self):
        user_data = self._build_data(self.train_set.matrix)
        item_data = self._build_data(self.train_set.matrix.T.tocsr())

        n_user = len(user_data[0])
        n_item = len(item_data[0])

        # R_user and R_item contain rating values
        R_user = user_data[1]
        R_item = item_data[1]

        if self.give_item_weight:
            item_weight = np.array([math.sqrt(len(i))
                                    for i in R_item], dtype=float)
            item_weight = (float(n_item) / item_weight.sum()) * item_weight
        else:
            item_weight = np.ones(n_item, dtype=float)

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

        for iter in range(self.max_iter):
            print("Iteration {}".format(iter + 1))
            tic = time.time()

            user_loss = np.zeros(n_user)
            for i in range(n_user):
                idx_item = user_data[0][i]
                V_i = self.V[idx_item]
                R_i = R_user[i]

                A = self.lambda_u * np.eye(self.dimension) + V_i.T.dot(V_i)
                B = (V_i * (np.tile(R_i, (self.dimension, 1)).T)).sum(0)
                self.U[i] = np.linalg.solve(A, B)

                user_loss[i] = -0.5 * self.lambda_u * np.dot(self.U[i], self.U[i])

            item_loss = np.zeros(n_item)
            for j in range(n_item):
                idx_user = item_data[0][j]
                U_j = self.U[idx_user]
                R_j = R_item[j]

                A = self.lambda_v * item_weight[j] * np.eye(self.dimension) + U_j.T.dot(U_j)
                B = (U_j * (np.tile(R_j, (self.dimension, 1)).T)).sum(0) \
                    + self.lambda_v * item_weight[j] * theta[j]
                self.V[j] = np.linalg.solve(A, B)

                item_loss[j] = -np.square(R_j - U_j.dot(self.V[j])).sum()

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
            toc = time.time()
            elapsed = toc - tic
            converge = abs((loss - history) / history)
            print("Loss: %.5f Elpased: %.4fs Converge: %.6f " % (loss, elapsed, converge))
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
