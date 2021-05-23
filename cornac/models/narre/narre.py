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

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, Input

from ...utils import get_rng
from ...utils.init_utils import uniform


class TextProcessor(keras.Model):
    def __init__(self, max_text_length, filters=64, kernel_sizes=[3], dropout_rate=0.5, name=''):
        super(TextProcessor, self).__init__(name=name)
        self.max_text_length = max_text_length
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.conv = []
        self.maxpool = []
        for kernel_size in kernel_sizes:
            self.conv.append(layers.Conv2D(self.filters, kernel_size=(1, kernel_size), use_bias=True, activation="relu"))
            self.maxpool.append(layers.MaxPooling2D(pool_size=(1, self.max_text_length - kernel_size + 1)))
        self.reshape = layers.Reshape(target_shape=(-1, self.filters * len(kernel_sizes)))
        self.dropout_rate = dropout_rate
        self.dropout = layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs, training=False):
        text = inputs
        pooled_outputs = []
        for conv, maxpool in zip(self.conv, self.maxpool):
            text_conv = conv(text)
            text_conv_maxpool = maxpool(text_conv)
            pooled_outputs.append(text_conv_maxpool)
        text_h = self.reshape(tf.concat(pooled_outputs, axis=-1))
        if training:
            text_h = self.dropout(text_h)
        return text_h


def get_data(batch_ids, train_set, max_text_length, by='user', max_num_review=None):
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences
    batch_reviews, batch_id_reviews, batch_num_reviews = [], [], []
    review_group = train_set.review_text.user_review if by == 'user' else train_set.review_text.item_review
    for idx in batch_ids:
        ids, review_ids = [], []
        for inc, (jdx, review_idx) in enumerate(review_group[idx].items()):
            if max_num_review is not None and inc == max_num_review:
                break
            ids.append(jdx)
            review_ids.append(review_idx)
        batch_id_reviews.append(ids)
        reviews = train_set.review_text.batch_seq(review_ids, max_length=max_text_length)
        batch_reviews.append(reviews)
        batch_num_reviews.append(len(reviews))
    batch_reviews = pad_sequences(batch_reviews, padding="post")
    batch_id_reviews = pad_sequences(batch_id_reviews, padding="post")
    batch_num_reviews = np.array(batch_num_reviews)
    return batch_reviews, batch_id_reviews, batch_num_reviews


class AddGlobalBias(keras.layers.Layer):

    def __init__(self, init_value=0.0, name="global_bias"):
        super(AddGlobalBias, self).__init__(name=name)
        self.init_value = init_value
      
    def build(self, input_shape):
        self.global_bias = self.add_weight(shape=1,
                               initializer=tf.keras.initializers.Constant(self.init_value),
                               trainable=True)

    def call(self, inputs):
        return inputs + self.global_bias

class Model:
    def __init__(self, n_users, n_items, vocab, global_mean, n_factors=32, embedding_size=100, id_embedding_size=32, attention_size=16, kernel_sizes=[3], n_filters=64, dropout_rate=0.5, max_text_length=50, pretrained_word_embeddings=None, verbose=False, seed=None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_vocab = vocab.size
        self.global_mean = global_mean
        self.n_factors = n_factors
        self.embedding_size = embedding_size
        self.id_embedding_size = id_embedding_size
        self.attention_size = attention_size
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.max_text_length = max_text_length
        self.verbose = verbose
        if seed is not None:
            self.rng = get_rng(seed)
            tf.random.set_seed(seed)

        embedding_matrix = uniform(shape=(self.n_vocab, self.embedding_size), low=-0.5, high=0.5, random_state=self.rng)
        embedding_matrix[:4, :] = np.zeros((4, self.embedding_size))
        if pretrained_word_embeddings is not None:
            oov_count = 0
            for word, idx in vocab.tok2idx.items():
                embedding_vector = pretrained_word_embeddings.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector
                else:
                    oov_count += 1
            if self.verbose:
                print("Number of OOV words: %d" % oov_count)

        embedding_matrix = initializers.Constant(embedding_matrix)
        i_user_id = Input(shape=(1,), dtype="int32", name="input_user_id")
        i_item_id = Input(shape=(1,), dtype="int32", name="input_item_id")
        i_user_review = Input(shape=(None, self.max_text_length), dtype="int32", name="input_user_review")
        i_item_review = Input(shape=(None, self.max_text_length), dtype="int32", name="input_item_review")
        i_user_iid_review = Input(shape=(None,), dtype="int32", name="input_user_iid_review")
        i_item_uid_review = Input(shape=(None,), dtype="int32", name="input_item_uid_review")
        i_user_num_reviews = Input(shape=(1,), dtype="int32", name="input_user_number_of_review")
        i_item_num_reviews = Input(shape=(1,), dtype="int32", name="input_item_number_of_review")

        l_user_review_embedding = layers.Embedding(self.n_vocab, self.embedding_size, embeddings_initializer=embedding_matrix, mask_zero=True, name="layer_user_review_embedding")
        l_item_review_embedding = layers.Embedding(self.n_vocab, self.embedding_size, embeddings_initializer=embedding_matrix, mask_zero=True, name="layer_item_review_embedding")
        l_user_iid_embedding = layers.Embedding(self.n_items, self.id_embedding_size, embeddings_initializer="uniform", name="user_iid_embedding")
        l_item_uid_embedding = layers.Embedding(self.n_users, self.id_embedding_size, embeddings_initializer="uniform", name="item_uid_embedding")
        l_user_embedding = layers.Embedding(self.n_users, self.id_embedding_size, embeddings_initializer="uniform", name="user_embedding")
        l_item_embedding = layers.Embedding(self.n_items, self.id_embedding_size, embeddings_initializer="uniform", name="item_embedding")
        user_bias = layers.Embedding(self.n_users, 1, embeddings_initializer=tf.initializers.Constant(0.1), name="user_bias")
        item_bias = layers.Embedding(self.n_items, 1, embeddings_initializer=tf.initializers.Constant(0.1), name="item_bias")

        user_text_processor = TextProcessor(self.max_text_length, filters=self.n_filters, kernel_sizes=self.kernel_sizes, dropout_rate=self.dropout_rate, name='user_text_processor')
        item_text_processor = TextProcessor(self.max_text_length, filters=self.n_filters, kernel_sizes=self.kernel_sizes, dropout_rate=self.dropout_rate, name='item_text_processor')

        user_review_h = user_text_processor(l_user_review_embedding(i_user_review), training=True)
        item_review_h = item_text_processor(l_item_review_embedding(i_item_review), training=True)
        a_user = layers.Dense(1, activation=None, use_bias=True)(
            layers.Dense(self.attention_size, activation="relu", use_bias=True)(
                tf.concat([user_review_h, l_user_iid_embedding(i_user_iid_review)], axis=-1)
            )
        )
        a_user_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_user_num_reviews, [-1]), maxlen=i_user_review.shape[1]), -1)
        user_attention = layers.Softmax(axis=1, name="user_attention")(a_user, a_user_masking)
        a_item = layers.Dense(1, activation=None, use_bias=True)(
            layers.Dense(self.attention_size, activation="relu", use_bias=True)(
                tf.concat([item_review_h, l_item_uid_embedding(i_item_uid_review)], axis=-1)
            )
        )
        a_item_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_item_num_reviews, [-1]), maxlen=i_item_review.shape[1]), -1)
        item_attention = layers.Softmax(axis=1, name="item_attention")(a_item, a_item_masking)

        Xu = layers.Dense(self.n_factors, use_bias=True, name="Xu")(
            layers.Dropout(rate=self.dropout_rate, name="user_Oi")(
                tf.reduce_sum(layers.Multiply()([user_attention, user_review_h]), 1)
            )
        )
        Yi = layers.Dense(self.n_factors, use_bias=True, name="Yi")(
            layers.Dropout(rate=self.dropout_rate, name="item_Oi")(
                tf.reduce_sum(layers.Multiply()([item_attention, item_review_h]), 1)
            )
        )

        h0 = layers.Multiply(name="h0")([
            layers.Add()([l_user_embedding(i_user_id), Xu]), layers.Add()([l_item_embedding(i_item_id), Yi])
        ])

        W1 = layers.Dense(1, activation=None, use_bias=False, name="W1")
        add_global_bias = AddGlobalBias(init_value=self.global_mean, name="global_bias")
        r = layers.Add(name="prediction")([
            W1(h0),
            user_bias(i_user_id),
            item_bias(i_item_id)
        ])
        r = add_global_bias(r)
        self.graph = keras.Model(inputs=[i_user_id, i_item_id, i_user_review, i_user_iid_review, i_user_num_reviews, i_item_review, i_item_uid_review, i_item_num_reviews], outputs=r)
        if self.verbose:
            self.graph.summary()

    def get_weights(self, train_set, batch_size=64, max_num_review=None):
        user_attention_review_pooling = keras.Model(inputs=[self.graph.get_layer('input_user_review').input, self.graph.get_layer('input_user_iid_review').input, self.graph.get_layer('input_user_number_of_review').input], outputs=self.graph.get_layer('Xu').output)
        item_attention_review_pooling = keras.Model(inputs=[self.graph.get_layer('input_item_review').input, self.graph.get_layer('input_item_uid_review').input, self.graph.get_layer('input_item_number_of_review').input], outputs=self.graph.get_layer('Yi').output)
        X = np.zeros((self.n_users, self.n_factors))
        Y = np.zeros((self.n_items, self.n_factors))
        for batch_users in train_set.user_iter(batch_size):
            user_reviews, user_iid_reviews, user_num_reviews = get_data(batch_users, train_set, self.max_text_length, by='user', max_num_review=max_num_review)
            Xu = user_attention_review_pooling([user_reviews, user_iid_reviews, user_num_reviews], training=False)
            X[batch_users] = Xu.numpy()
        for batch_items in train_set.item_iter(batch_size):
            item_reviews, item_uid_reviews, item_num_reviews = get_data(batch_items, train_set, self.max_text_length, by='item', max_num_review=max_num_review)
            Yi = item_attention_review_pooling([item_reviews, item_uid_reviews, item_num_reviews], training=False)
            Y[batch_items] = Yi.numpy()
        W1 = self.graph.get_layer('W1').get_weights()[0]
        user_embedding = self.graph.get_layer('user_embedding').get_weights()[0]
        item_embedding = self.graph.get_layer('item_embedding').get_weights()[0]
        bu = self.graph.get_layer('user_bias').get_weights()[0]
        bi = self.graph.get_layer('item_bias').get_weights()[0]
        mu = self.graph.get_layer('global_bias').get_weights()[0][0]
        if self.verbose:
            print('global bias =', mu)
        return X, Y, W1, user_embedding, item_embedding, bu, bi, mu
