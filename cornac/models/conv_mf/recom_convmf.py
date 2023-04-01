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

import os
import time
import math

import numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.init_utils import xavier_uniform


class ConvMF(Recommender):
    """
    Parameters
    ----------
    k: int, optional, default: 50
        The dimension of the user and item latent factors.

    n_epochs: int, optional, default: 50
        Maximum number of epochs for training.

    cnn_epochs: int, optional, default: 5
        Number of epochs for optimizing the CNN for each overall training epoch.
    
    cnn_bs: int, optional, default: 128
        Batch size for optimizing CNN.
        
    cnn_lr: float, optional, default: 0.001
        Learning rate for optimizing CNN.

    lambda_u: float, optional, default: 1.0
        The regularization hyper-parameter for user latent factor.

    lambda_v: float, optional, default: 100.0
        The regularization hyper-parameter for item latent factor.

    emb_dim: int, optional, default: 200
        The embedding size of each word. One word corresponds with [1 x emb_dim] vector in the embedding space

    max_len: int, optional, default 300
        The maximum length of item's document

    filter_sizes: list, optional, default: [3, 4, 5]
        The length of filters in convolutional layer

    num_filters: int, optional, default: 100
        The number of filters in convolutional layer
        
    hidden_dim: int, optional, default: 200
        The dimension of hidden layer after the pooling of all convolutional layers

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

    def __init__(
        self,
        name="ConvMF",
        k=50,
        n_epochs=50,
        cnn_epochs=5,
        cnn_bs=128,
        cnn_lr=0.001,
        lambda_u=1,
        lambda_v=100,
        emb_dim=200,
        max_len=300,
        filter_sizes=[3, 4, 5],
        num_filters=100,
        hidden_dim=200,
        dropout_rate=0.2,
        give_item_weight=True,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.give_item_weight = give_item_weight
        self.n_epochs = n_epochs
        self.cnn_bs = cnn_bs
        self.cnn_lr = cnn_lr
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.k = k
        self.dropout_rate = dropout_rate
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.name = name
        self.verbose = verbose
        self.cnn_epochs = cnn_epochs
        self.seed = seed

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)
        self.V = self.init_params.get("V", None)
        self.W = self.init_params.get("W", None)

    def _init(self):
        rng = get_rng(self.seed)
        n_users, n_items = self.train_set.num_users, self.train_set.num_items
        vocab_size = self.train_set.item_text.vocab.size

        if self.U is None:
            self.U = xavier_uniform((n_users, self.k), rng)
        if self.V is None:
            self.V = xavier_uniform((n_items, self.k), rng)
        if self.W is None:
            self.W = xavier_uniform((vocab_size, self.emb_dim), rng)

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        self._init()

        if self.trainable:
            self._fit_convmf()

        return self

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
            item_weight = np.array([math.sqrt(len(i)) for i in R_item], dtype=float)
            item_weight = (float(n_item) / item_weight.sum()) * item_weight
        else:
            item_weight = np.ones(n_item, dtype=float)

        # Initialize cnn module
        import tensorflow.compat.v1 as tf
        from .convmf import CNN_module

        tf.disable_eager_execution()

        # less verbose TF
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.logging.set_verbosity(tf.logging.ERROR)

        tf.set_random_seed(self.seed)
        cnn_module = CNN_module(
            output_dimension=self.k,
            dropout_rate=self.dropout_rate,
            emb_dim=self.emb_dim,
            max_len=self.max_len,
            filter_sizes=self.filter_sizes,
            num_filters=self.num_filters,
            hidden_dim=self.hidden_dim,
            seed=self.seed,
            init_W=self.W,
            learning_rate=self.cnn_lr,
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())  # init variable

        document = self.train_set.item_text.batch_seq(
            np.arange(n_item), max_length=self.max_len
        )

        feed_dict = {cnn_module.model_input: document}
        theta = sess.run(cnn_module.model_output, feed_dict=feed_dict)

        endure = 3
        converge_threshold = 0.01
        history = 1e-50
        loss = 0

        for epoch in range(1, self.n_epochs + 1):
            if self.verbose:
                print("Epoch: {}/{}".format(epoch, self.n_epochs))

            tic = time.time()

            user_loss = 0.0
            for i in range(n_user):
                idx_item = user_data[0][i]
                V_i = self.V[idx_item]
                R_i = R_user[i]

                A = self.lambda_u * np.eye(self.k) + V_i.T.dot(V_i)
                B = (V_i * (np.tile(R_i, (self.k, 1)).T)).sum(0)
                self.U[i] = np.linalg.solve(A, B)

                user_loss += self.lambda_u * np.dot(self.U[i], self.U[i])

            item_loss = 0.0
            for j in range(n_item):
                idx_user = item_data[0][j]
                U_j = self.U[idx_user]
                R_j = R_item[j]

                A = self.lambda_v * item_weight[j] * np.eye(self.k) + U_j.T.dot(U_j)
                B = (U_j * (np.tile(R_j, (self.k, 1)).T)).sum(
                    0
                ) + self.lambda_v * item_weight[j] * theta[j]
                self.V[j] = np.linalg.solve(A, B)

                item_loss += np.square(R_j - U_j.dot(self.V[j])).sum()

            loop = trange(
                self.cnn_epochs, desc="Optimizing CNN", disable=not self.verbose
            )
            for _ in loop:
                for batch_ids in self.train_set.item_iter(
                    batch_size=self.cnn_bs, shuffle=True
                ):
                    batch_seq = self.train_set.item_text.batch_seq(
                        batch_ids, max_length=self.max_len
                    )
                    feed_dict = {
                        cnn_module.model_input: batch_seq,
                        cnn_module.v: self.V[batch_ids],
                        cnn_module.sample_weight: item_weight[batch_ids],
                    }

                    sess.run([cnn_module.optimizer], feed_dict=feed_dict)

            feed_dict = {
                cnn_module.model_input: document,
                cnn_module.v: self.V,
                cnn_module.sample_weight: item_weight,
            }
            theta, cnn_loss = sess.run(
                [cnn_module.model_output, cnn_module.weighted_loss], feed_dict=feed_dict
            )

            loss = 0.5 * (user_loss + item_loss + self.lambda_v * cnn_loss)

            toc = time.time()
            elapsed = toc - tic
            converge = abs((loss - history) / history)

            if self.verbose:
                print(
                    "Loss: %.5f Elapsed: %.4fs Converge: %.6f "
                    % (loss, elapsed, converge)
                )

            history = loss
            if converge < converge_threshold:
                endure -= 1
                if endure == 0:
                    break

        tf.reset_default_graph()

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            known_item_scores = self.V.dot(self.U[user_idx, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = self.V[item_idx, :].dot(self.U[user_idx, :])

            return user_pred
