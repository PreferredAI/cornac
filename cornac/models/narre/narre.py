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
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from ...utils import get_rng
from ...utils.init_utils import uniform


class TextProcessor(keras.Model):
    def __init__(self, max_text_length, filters=64, kernel_sizes=[3], dropout_rate=0.5, name='', **kwargs):
        super(TextProcessor, self).__init__(name=name, **kwargs)
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
    batch_reviews = pad_sequences(batch_reviews, maxlen=max_num_review, padding="post")
    batch_id_reviews = pad_sequences(batch_id_reviews, maxlen=max_num_review, padding="post")
    batch_num_reviews = np.array(batch_num_reviews)
    return batch_reviews, batch_id_reviews, batch_num_reviews


class AddGlobalBias(keras.layers.Layer):

    def __init__(self, init_value=0.0, name="global_bias"):
        super(AddGlobalBias, self).__init__(name=name)
        self.init_value = init_value
      
    def build(self, input_shape):
        self.global_bias = self.add_weight(shape=1,
                               initializer=tf.keras.initializers.Constant(self.init_value),
                               trainable=True, name="add_weight")

    def call(self, inputs):
        return inputs + self.global_bias

class Model(keras.Model):
    def __init__(self, n_users, n_items, n_vocab, embedding_matrix, global_mean, n_factors=32, embedding_size=100, id_embedding_size=32, attention_size=16, kernel_sizes=[3], n_filters=64, dropout_rate=0.5, max_text_length=50):
        super().__init__()
        self.l_user_review_embedding = layers.Embedding(n_vocab, embedding_size, embeddings_initializer=embedding_matrix, mask_zero=True, name="layer_user_review_embedding")
        self.l_item_review_embedding = layers.Embedding(n_vocab, embedding_size, embeddings_initializer=embedding_matrix, mask_zero=True, name="layer_item_review_embedding")
        self.l_user_iid_embedding = layers.Embedding(n_items, id_embedding_size, embeddings_initializer="uniform", name="user_iid_embedding")
        self.l_item_uid_embedding = layers.Embedding(n_users, id_embedding_size, embeddings_initializer="uniform", name="item_uid_embedding")
        self.l_user_embedding = layers.Embedding(n_users, id_embedding_size, embeddings_initializer="uniform", name="user_embedding")
        self.l_item_embedding = layers.Embedding(n_items, id_embedding_size, embeddings_initializer="uniform", name="item_embedding")
        self.user_bias = layers.Embedding(n_users, 1, embeddings_initializer=tf.initializers.Constant(0.1), name="user_bias")
        self.item_bias = layers.Embedding(n_items, 1, embeddings_initializer=tf.initializers.Constant(0.1), name="item_bias")
        self.user_text_processor = TextProcessor(max_text_length, filters=n_filters, kernel_sizes=kernel_sizes, dropout_rate=dropout_rate, name='user_text_processor')
        self.item_text_processor = TextProcessor(max_text_length, filters=n_filters, kernel_sizes=kernel_sizes, dropout_rate=dropout_rate, name='item_text_processor')
        self.a_user = keras.models.Sequential([
            layers.Dense(attention_size, activation="relu", use_bias=True),
            layers.Dense(1, activation=None, use_bias=True)
        ])
        self.user_attention = layers.Softmax(axis=1, name="user_attention")
        self.a_item = keras.models.Sequential([
            layers.Dense(attention_size, activation="relu", use_bias=True),
            layers.Dense(1, activation=None, use_bias=True)
        ])
        self.item_attention = layers.Softmax(axis=1, name="item_attention")
        self.user_Oi_dropout = layers.Dropout(rate=dropout_rate, name="user_Oi")
        self.Xu = layers.Dense(n_factors, use_bias=True, name="Xu")
        self.item_Oi_dropout = layers.Dropout(rate=dropout_rate, name="item_Oi")
        self.Yi = layers.Dense(n_factors, use_bias=True, name="Yi")

        self.W1 = layers.Dense(1, activation=None, use_bias=False, name="W1")
        self.add_global_bias = AddGlobalBias(init_value=global_mean, name="global_bias")

    def call(self, inputs, training=None):
        i_user_id, i_item_id, i_user_review, i_user_iid_review, i_user_num_reviews, i_item_review, i_item_uid_review, i_item_num_reviews = inputs
        user_review_h = self.user_text_processor(self.l_user_review_embedding(i_user_review), training=training)
        a_user = self.a_user(tf.concat([user_review_h, self.l_user_iid_embedding(i_user_iid_review)], axis=-1))
        a_user_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_user_num_reviews, [-1]), maxlen=i_user_review.shape[1]), -1)
        user_attention = self.user_attention(a_user, a_user_masking)
        user_Oi = self.user_Oi_dropout(tf.reduce_sum(tf.multiply(user_attention, user_review_h), 1), training=training)
        Xu = self.Xu(user_Oi)
        item_review_h = self.item_text_processor(self.l_item_review_embedding(i_item_review), training=training)
        a_item = self.a_item(tf.concat([item_review_h, self.l_item_uid_embedding(i_item_uid_review)], axis=-1))
        a_item_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_item_num_reviews, [-1]), maxlen=i_item_review.shape[1]), -1)
        item_attention = self.item_attention(a_item, a_item_masking)
        item_Oi = self.item_Oi_dropout(tf.reduce_sum(tf.multiply(item_attention, item_review_h), 1), training=training)
        Yi = self.Yi(item_Oi)
        h0 = tf.multiply(tf.add(self.l_user_embedding(i_user_id), Xu), tf.add(self.l_item_embedding(i_item_id), Yi))
        r = self.add_global_bias(
            tf.add_n([
                self.W1(h0),
                self.user_bias(i_user_id),
                self.item_bias(i_item_id)
            ])
        )
        # import pdb; pdb.set_trace()
        return r

class NARREModel:
    def __init__(self, n_users, n_items, vocab, global_mean, n_factors=32, embedding_size=100, id_embedding_size=32, attention_size=16, kernel_sizes=[3], n_filters=64, dropout_rate=0.5, max_text_length=50, max_num_review=32, pretrained_word_embeddings=None, verbose=False, seed=None):
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
        self.max_num_review = max_num_review
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
        self.graph = Model(
            self.n_users, self.n_items, self.n_vocab, embedding_matrix, self.global_mean,
            self.n_factors, self.embedding_size, self.id_embedding_size, self.attention_size,
            self.kernel_sizes, self.n_filters, self.dropout_rate, self.max_text_length
        )

    def get_weights(self, train_set, batch_size=64):
        X = np.zeros((self.n_users, self.n_factors))
        Y = np.zeros((self.n_items, self.n_factors))
        for batch_users in train_set.user_iter(batch_size):
            i_user_review, i_user_iid_review, i_user_num_reviews = get_data(batch_users, train_set, self.max_text_length, by='user', max_num_review=self.max_num_review)
            user_review_embedding = self.graph.l_user_review_embedding(i_user_review)
            user_review_h = self.graph.user_text_processor(user_review_embedding, training=False)
            a_user = self.graph.a_user(tf.concat([user_review_h, self.graph.l_user_iid_embedding(i_user_iid_review)], axis=-1))
            a_user_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_user_num_reviews, [-1]), maxlen=i_user_review.shape[1]), -1)
            user_attention = self.graph.user_attention(a_user, a_user_masking)
            user_Oi = tf.reduce_sum(tf.multiply(user_attention, user_review_h), 1)
            Xu = self.graph.Xu(user_Oi)
            X[batch_users] = Xu.numpy()
        for batch_items in train_set.item_iter(batch_size):
            i_item_review, i_item_uid_review, i_item_num_reviews = get_data(batch_items, train_set, self.max_text_length, by='item', max_num_review=self.max_num_review)
            item_review_embedding = self.graph.l_item_review_embedding(i_item_review)
            item_review_h = self.graph.item_text_processor(item_review_embedding, training=False)
            a_item = self.graph.a_item(tf.concat([item_review_h, self.graph.l_item_uid_embedding(i_item_uid_review)], axis=-1))
            a_item_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_item_num_reviews, [-1]), maxlen=i_item_review.shape[1]), -1)
            item_attention = self.graph.item_attention(a_item, a_item_masking)
            item_Oi = tf.reduce_sum(tf.multiply(item_attention, item_review_h), 1)
            Yi = self.graph.Yi(item_Oi)
            Y[batch_items] = Yi.numpy()
        W1 = self.graph.W1.get_weights()[0]
        user_embedding = self.graph.l_user_embedding.get_weights()[0]
        item_embedding = self.graph.l_item_embedding.get_weights()[0]
        bu = self.graph.user_bias.get_weights()[0]
        bi = self.graph.item_bias.get_weights()[0]
        mu = self.graph.add_global_bias.get_weights()[0][0]
        return X, Y, W1, user_embedding, item_embedding, bu, bi, mu
