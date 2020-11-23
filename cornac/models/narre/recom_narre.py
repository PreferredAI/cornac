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
import numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng, estimate_batches
from ...utils.init_utils import xavier_uniform


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class NARRE(Recommender):
    """Neural Attentional Rating Regression with Review-level Explanations

    Parameters
    ----------
    name: string, default: 'NARRE'
        The name of the recommender model.

    embedding_size: int, default: 100
        Word embedding size

    id_embedding_size: int, default: 32
        User/item review id embedding size

    n_factors: int, default: 32
        The dimension of the user/item's latent factors.

    attention_size: int, default: 16
        Attention size

    kernel_sizes: list, default: [3]
        List of kernel sizes of conv2d

    n_filters: int, default: 64
        Number of filters

    dropout_rate: float, default: 0.5
        Dropout rate of neural network dense layers

    max_review_length: int, default: 50
        Maximum number of tokens in a review instance

    batch_size: int, default: 64
        Batch size

    max_iter: int, default: 10
        Max number of training epochs

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, pretrained_word_embeddings could be initialized here, e.g., init_params={'pretrained_word_embeddings': pretrained_word_embeddings}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Chen, C., Zhang, M., Liu, Y., & Ma, S. (2018, April). Neural attentional rating regression with review-level explanations. In Proceedings of the 2018 World Wide Web Conference (pp. 1583-1592).
    """

    def __init__(
        self,
        name="NARRE",
        embedding_size=100,
        id_embedding_size=32,
        n_factors=32,
        attention_size=16,
        kernel_sizes=[3],
        n_filters=64,
        dropout_rate=0.5,
        max_review_length=50,
        batch_size=64,
        max_iter=10,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.seed = seed
        self.rng = get_rng(seed)
        self.embedding_size = embedding_size
        self.id_embedding_size = id_embedding_size
        self.n_factors = n_factors
        self.attention_size = attention_size
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.max_review_length = max_review_length
        self.batch_size = batch_size
        self.max_iter = max_iter
        # Init params if provided
        self.init_params = {} if init_params is None else init_params

    def _init(self):
        self.n_users, self.n_items = self.train_set.num_users, self.train_set.num_items
        self.n_vocab = self.train_set.review_text.vocab.size
        self.pretrained_word_embeddings = self.init_params.get('pretrained_word_embeddings')
        self._init_word_embedding_matrix()

    def _init_word_embedding_matrix(self):
        self.embedding_matrix = np.random.uniform(-0.5, 0.5, (self.n_vocab, self.embedding_size))
        self.embedding_matrix[:4, :] = np.zeros((4, self.embedding_size))
        if self.pretrained_word_embeddings is not None:
            oov_count = 0
            for word, idx in self.train_set.review_text.vocab.tok2idx.items():
                embedding_vector = self.pretrained_word_embeddings.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[idx] = embedding_vector
                else:
                    oov_count += 1
            if self.verbose:
                print("Number of OOV words: %d" % oov_count)

    def _init_model(self):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import Input, layers, initializers, Model
        from .narre import TextProcessor
        tf.random.set_seed(self.seed)
        self.i_user_id = Input(shape=(1,), dtype="int32", name="input_user_id")
        self.i_item_id = Input(shape=(1,), dtype="int32", name="input_item_id")
        self.i_user_review = Input(shape=(None, self.max_review_length), dtype="int32", name="input_user_review")
        self.i_item_review = Input(shape=(None, self.max_review_length), dtype="int32", name="input_item_review")
        self.i_user_iid_review = Input(shape=(None,), dtype="int32", name="input_user_iid_review")
        self.i_item_uid_review = Input(shape=(None,), dtype="int32", name="input_item_iid_review")
        self.l_user_review_embedding = layers.Embedding(self.n_vocab, self.embedding_size, embeddings_initializer=initializers.Constant(self.embedding_matrix), name="layer_user_review_embedding")
        self.l_item_review_embedding = layers.Embedding(self.n_vocab, self.embedding_size, embeddings_initializer=initializers.Constant(self.embedding_matrix), name="layer_item_review_embedding")
        self.l_user_iid_embedding = layers.Embedding(self.n_items, self.id_embedding_size, embeddings_initializer="uniform", name="user_iid_embedding")
        self.l_item_uid_embedding = layers.Embedding(self.n_users, self.id_embedding_size, embeddings_initializer="uniform", name="item_uid_embedding")
        self.l_user_embedding = layers.Embedding(self.n_users, self.id_embedding_size, embeddings_initializer="uniform", name="user_embedding")
        self.l_item_embedding = layers.Embedding(self.n_items, self.id_embedding_size, embeddings_initializer="uniform", name="item_embedding")
        self.user_bias = layers.Embedding(self.n_users, 1, embeddings_initializer=tf.initializers.Constant(0.1))
        self.item_bias = layers.Embedding(self.n_items, 1, embeddings_initializer=tf.initializers.Constant(0.1))
        self.global_bias = keras.backend.constant(self.train_set.global_mean, shape=(1,), name="global_mean")

        self.user_text_processor = TextProcessor(self.max_review_length, filters=self.n_filters, kernel_sizes=self.kernel_sizes, dropout_rate=self.dropout_rate)
        self.item_text_processor = TextProcessor(self.max_review_length, filters=self.n_filters, kernel_sizes=self.kernel_sizes, dropout_rate=self.dropout_rate)
        user_review_h = self.user_text_processor(self.l_user_review_embedding(self.i_user_review), training=self.trainable)
        item_review_h = self.item_text_processor(self.l_item_review_embedding(self.i_item_review), training=self.trainable)

        self.user_attention = layers.Softmax(name="user_attention")(
            layers.Dense(self.n_filters, activation=None, use_bias=True)(
                layers.Dense(self.attention_size, activation="relu", use_bias=True)(
                    tf.concat([user_review_h, self.l_user_iid_embedding(self.i_user_iid_review)], axis=-1)
                )
            )
        )
        self.item_attention = layers.Softmax(name="item_attention")(
            layers.Dense(self.n_filters, activation=None, use_bias=True)(
                layers.Dense(self.attention_size, activation="relu", use_bias=True)(
                    tf.concat([item_review_h, self.l_item_uid_embedding(self.i_item_uid_review)], axis=-1)
                )
            )
        )

        u = layers.Dropout(rate=self.dropout_rate, name="user_Oi")(tf.reduce_sum(layers.Multiply()([self.user_attention, user_review_h]), 1))
        i = layers.Dropout(rate=self.dropout_rate, name="item_Oi")(tf.reduce_sum(layers.Multiply()([self.item_attention, item_review_h]), 1))
        self.Xu = layers.Dense(self.n_factors, use_bias=True, name="Xu")(u)
        self.Yi = layers.Dense(self.n_factors, use_bias=True, name="Yi")(i)

        self.user_attention_review_pooling = Model(inputs=[self.i_user_id, self.i_user_review, self.i_user_iid_review], outputs=self.Xu)
        self.item_attention_review_pooling = Model(inputs=[self.i_item_id, self.i_item_review, self.i_item_uid_review], outputs=self.Yi)

        h0 = layers.Multiply(name="h0")([
            layers.Add()([self.l_user_embedding(self.i_user_id), self.Xu]), layers.Add()([self.l_item_embedding(self.i_item_id), self.Yi])
        ])

        self.W1 = layers.Dense(1, activation=None, use_bias=False)
        r = layers.Add(name="prediction")([
            self.W1(h0),
            self.user_bias(self.i_user_id),
            self.item_bias(self.i_item_id),
            self.global_bias,
        ])
        self.model = Model(inputs=[self.i_user_id, self.i_item_id, self.i_user_review, self.i_user_iid_review, self.i_item_review, self.i_item_uid_review], outputs=r)
        if self.verbose:
            self.model.summary()
        self.loss = keras.losses.MeanSquaredError()
        self.optimizer = keras.optimizers.Adam()
        self.train_loss = keras.metrics.Mean(name="loss")

    def _get_data(self, batch_ids, by='user'):
        from tensorflow.python.keras.preprocessing.sequence import pad_sequences
        batch_reviews, batch_id_reviews = [], []
        review_group = self.train_set.review_text.user_review if by == 'user' else self.train_set.review_text.item_review
        for idx in batch_ids:
            ids, review_ids = [], []
            for jdx, review_idx in review_group[idx].items():
                ids.append(jdx)
                review_ids.append(review_idx)
            batch_id_reviews.append(ids)
            reviews = self.train_set.review_text.batch_seq(review_ids, max_length=self.max_review_length)
            batch_reviews.append(reviews)
        batch_reviews = pad_sequences(batch_reviews, padding="post")
        batch_id_reviews = pad_sequences(batch_id_reviews, padding="post")
        return batch_reviews, batch_id_reviews

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
            if not hasattr(self, "model"):
                self._init_model()
            self._fit_narre()

        return self

    def _fit_narre(self):
        import tensorflow as tf
        loop = trange(self.max_iter, disable=not self.verbose)
        for _ in loop:
            self.train_loss.reset_states()
            for i, (batch_users, batch_items, batch_ratings) in enumerate(self.train_set.uir_iter(self.batch_size, shuffle=True)):
                user_reviews, user_iid_reviews = self._get_data(batch_users, by='user')
                item_reviews, item_uid_reviews = self._get_data(batch_items, by='item')
                with tf.GradientTape() as tape:
                    predictions = self.model(
                        [batch_users, batch_items, user_reviews, user_iid_reviews, item_reviews, item_uid_reviews],
                        training=True,
                    )
                    loss = self.loss(batch_ratings, predictions)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                self.train_loss(loss)
                if i % 10 == 0:
                    loop.set_postfix(loss=self.train_loss.result().numpy())
        loop.close()

        # save weights for predictions
        self.X = np.zeros((self.n_users, self.n_factors))
        self.Y = np.zeros((self.n_items, self.n_factors))
        for batch_users in self.train_set.user_iter(self.batch_size):
            user_reviews, user_iid_reviews = self._get_data(batch_users, by='user')
            Xu = self.user_attention_review_pooling([batch_users, user_reviews, user_iid_reviews], training=False)
            self.X[batch_users] = Xu.numpy()

        for batch_items in self.train_set.item_iter(self.batch_size):
            item_reviews, item_uid_reviews = self._get_data(batch_items, by='item')
            Yi = self.item_attention_review_pooling([batch_items, item_reviews, item_uid_reviews], training=False)
            self.Y[batch_items] = Yi.numpy()

        self.W1_weights = self.W1.get_weights()[0]
        self.user_embedding = self.l_user_embedding.get_weights()[0]
        self.item_embedding = self.l_item_embedding.get_weights()[0]
        self.bu = self.user_bias.get_weights()[0]
        self.bi = self.item_bias.get_weights()[0]
        self.mu = self.global_bias.numpy()

        if self.verbose:
            print("Learning completed!")

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
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
            h0 = (self.user_embedding[user_idx] + self.X[user_idx]) * (self.item_embedding + self.Y)
            known_item_scores = h0.dot(self.W1_weights) + self.bu[user_idx] + self.bi + self.mu
            return known_item_scores.ravel()
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            h0 = (self.user_embedding[user_idx] + self.X[user_idx]) * (self.item_embedding[item_idx] + self.Y[item_idx])
            known_item_score = h0.dot(self.W1_weights) + self.bu[user_idx] + self.bi[item_idx] + self.mu
            return known_item_score