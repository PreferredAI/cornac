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

from ...data.text import Vocabulary
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Model(object):
    def __init__(
        self,
        review_num_u,
        review_num_i,
        review_len_u,
        review_len_i,
        n_users,
        n_items,
        n_latent,
        id_embedding_size,
        word_embedding_size,
        attention_size,
        filter_sizes,
        n_filters,
        user_vocabulary: Vocabulary,
        item_vocabulary: Vocabulary,
        l2_reg_lambda=0.001,
        learning_rate=0.002,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        pretrained_word_embeddings=None,
    ):
        self.review_num_u = review_num_u
        self.review_num_i = review_num_i
        self.review_len_u = review_len_u
        self.review_len_i = review_len_i
        self.n_users = n_users
        self.n_items = n_items
        self.user_vocabulary = user_vocabulary
        self.n_user_vocab = user_vocabulary.size
        self.item_vocabulary = item_vocabulary
        self.n_item_vocab = item_vocabulary.size
        self.pretrained_word_embeddings = {
        } if pretrained_word_embeddings is None else pretrained_word_embeddings
        self.n_latent = n_latent
        self.id_embedding_size = id_embedding_size
        self.word_embedding_size = word_embedding_size
        self.attention_size = attention_size
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._build_graph()

    def _build_graph(self):
        self.input_u = tf.placeholder(
            dtype=tf.int32,
            shape=(None, None, None),
            name="input_u"
        )
        self.input_i = tf.placeholder(
            dtype=tf.int32,
            shape=(None, None, None),
            name="input_i"
        )
        self.input_reuid = tf.placeholder(
            dtype=tf.int32,
            shape=(None, None),
            name="input_reuid"
        )
        self.input_reiid = tf.placeholder(
            dtype=tf.int32,
            shape=(None, None),
            name="input_reiid"
        )
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        iidW = tf.Variable(
            tf.random_uniform(
                [self.n_items + 2, self.id_embedding_size], -0.1, 0.1),
            name="iidW",
        )
        uidW = tf.Variable(
            tf.random_uniform(
                [self.n_users + 2, self.id_embedding_size], -0.1, 0.1),
            name="uidW",
        )

        # assign pretrained word embeddings
        initWu = np.random.uniform(
            -1.0, 1.0, (self.n_user_vocab, self.word_embedding_size)
        )
        initWi = np.random.uniform(
            -1.0, 1.0, (self.n_item_vocab, self.word_embedding_size)
        )

        if self.pretrained_word_embeddings:
            for word, idx in self.user_vocabulary.tok2idx.items():
                if word in self.pretrained_word_embeddings:
                    initWu[idx] = self.pretrained_word_embeddings[word]
            for word, idx in self.item_vocabulary.tok2idx.items():
                if word in self.pretrained_word_embeddings:
                    initWi[idx] = self.pretrained_word_embeddings[word]

        l2_loss = tf.constant(0.0)
        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(initWu, name="W1", dtype=tf.float32)
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(initWi, name="W2", dtype=tf.float32)
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)

        pooled_outputs_u = []
        for filter_size in self.filter_sizes:
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [
                    filter_size,
                    self.word_embedding_size,
                    1,
                    self.n_filters,
                ]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[self.n_filters]), name="b")
                self.embedded_users = tf.reshape(
                    self.embedded_users,
                    [-1, self.review_len_u, self.word_embedding_size, 1],
                )
                conv = tf.nn.conv2d(
                    self.embedded_users,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv",
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.review_len_u - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool",
                )
                pooled_outputs_u.append(pooled)
        n_filters_total = self.n_filters * len(self.filter_sizes)
        self.h_pool_u = tf.concat(pooled_outputs_u, 3)

        self.h_pool_flat_u = tf.reshape(
            self.h_pool_u, [-1, self.review_num_u, n_filters_total]
        )

        pooled_outputs_i = []

        for filter_size in self.filter_sizes:
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [
                    filter_size,
                    self.word_embedding_size,
                    1,
                    self.n_filters,
                ]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[self.n_filters]), name="b")
                self.embedded_items = tf.reshape(
                    self.embedded_items,
                    [-1, self.review_len_i, self.word_embedding_size, 1],
                )
                conv = tf.nn.conv2d(
                    self.embedded_items,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv",
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.review_len_i - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool",
                )
                pooled_outputs_i.append(pooled)
        n_filters_total = self.n_filters * len(self.filter_sizes)
        self.h_pool_i = tf.concat(pooled_outputs_i, 3)
        self.h_pool_flat_i = tf.reshape(
            self.h_pool_i, [-1, self.review_num_i, n_filters_total]
        )

        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)

        with tf.name_scope("attention"):
            Wau = tf.Variable(
                tf.random_uniform(
                    [n_filters_total, self.attention_size], -0.1, 0.1),
                name="Wau",
            )
            Wru = tf.Variable(
                tf.random_uniform(
                    [self.id_embedding_size, self.attention_size], -0.1, 0.1
                ),
                name="Wru",
            )
            Wpu = tf.Variable(
                tf.random_uniform([self.attention_size, 1], -0.1, 0.1), name="Wpu"
            )
            bau = tf.Variable(tf.constant(
                0.1, shape=[self.attention_size]), name="bau")
            bbu = tf.Variable(tf.constant(0.1, shape=[1]), name="bbu")
            self.iid_a = tf.nn.relu(
                tf.nn.embedding_lookup(iidW, self.input_reuid))
            self.u_j = (
                tf.einsum(
                    "ajk,kl->ajl",
                    tf.nn.relu(
                        tf.einsum("ajk,kl->ajl", self.h_drop_u, Wau)
                        + tf.einsum("ajk,kl->ajl", self.iid_a, Wru)
                        + bau
                    ),
                    Wpu,
                )
                + bbu
            )  # None*u_len*1
            self.u_a = tf.nn.softmax(self.u_j, 1)  # none*u_len*1
            Wai = tf.Variable(
                tf.random_uniform(
                    [n_filters_total, self.attention_size], -0.1, 0.1),
                name="Wai",
            )
            Wri = tf.Variable(
                tf.random_uniform(
                    [self.id_embedding_size, self.attention_size], -0.1, 0.1
                ),
                name="Wri",
            )
            Wpi = tf.Variable(
                tf.random_uniform([self.attention_size, 1], -0.1, 0.1), name="Wpi"
            )
            bai = tf.Variable(tf.constant(
                0.1, shape=[self.attention_size]), name="bai")
            bbi = tf.Variable(tf.constant(0.1, shape=[1]), name="bbi")
            self.uid_a = tf.nn.relu(
                tf.nn.embedding_lookup(uidW, self.input_reuid))
            self.i_j = (
                tf.einsum(
                    "ajk,kl->ajl",
                    tf.nn.relu(
                        tf.einsum("ajk,kl->ajl", self.h_drop_i, Wai)
                        + tf.einsum("ajk,kl->ajl", self.uid_a, Wri)
                        + bai
                    ),
                    Wpi,
                )
                + bbi
            )
            self.i_a = tf.nn.softmax(self.i_j, 1)  # none*len*1
            l2_loss += tf.nn.l2_loss(Wau)
            l2_loss += tf.nn.l2_loss(Wru)
            l2_loss += tf.nn.l2_loss(Wri)
            l2_loss += tf.nn.l2_loss(Wai)

        with tf.name_scope("add_reviews"):
            self.u_feas = tf.reduce_sum(
                tf.multiply(self.u_a, self.h_drop_u), 1)
            self.u_feas = tf.nn.dropout(self.u_feas, self.dropout_keep_prob)
            self.i_feas = tf.reduce_sum(
                tf.multiply(self.i_a, self.h_drop_i), 1)
            self.i_feas = tf.nn.dropout(self.i_feas, self.dropout_keep_prob)

        with tf.name_scope("get_fea"):
            uidmf = tf.Variable(
                tf.random_uniform(
                    [self.n_users + 2, self.id_embedding_size], -0.1, 0.1
                ),
                name="uidmf",
            )
            self.uid = tf.nn.embedding_lookup(uidmf, self.input_uid)
            self.uid = tf.reshape(self.uid, [-1, self.id_embedding_size])
            Wu = tf.Variable(
                tf.random_uniform([n_filters_total, self.n_latent], -0.1, 0.1),
                name="Wu",
            )
            bu = tf.Variable(tf.constant(
                0.1, shape=[self.n_latent]), name="bu")
            self.u_feas = tf.matmul(self.u_feas, Wu) + self.uid + bu

            iidmf = tf.Variable(
                tf.random_uniform(
                    [self.n_items + 2, self.id_embedding_size], -0.1, 0.1
                ),
                name="iidmf",
            )
            self.iid = tf.nn.embedding_lookup(iidmf, self.input_iid)
            self.iid = tf.reshape(self.iid, [-1, self.id_embedding_size])
            Wi = tf.Variable(
                tf.random_uniform([n_filters_total, self.n_latent], -0.1, 0.1),
                name="Wi",
            )
            bi = tf.Variable(tf.constant(
                0.1, shape=[self.n_latent]), name="bi")
            self.i_feas = tf.matmul(self.i_feas, Wi) + self.iid + bi

        with tf.name_scope("ncf"):
            self.FM = tf.multiply(self.u_feas, self.i_feas)  # h0
            self.FM = tf.nn.relu(self.FM)

            self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)

            self.Wmul = tf.Variable(
                tf.random_uniform([self.n_latent, 1], -0.1, 0.1), name="wmul"
            )

            self.mul = tf.matmul(self.FM, self.Wmul)
            self.score = tf.reduce_sum(self.mul, 1, keep_dims=True)

            self.uidW2 = tf.Variable(
                tf.constant(0.1, shape=[self.n_users + 2]), name="uidW2"
            )
            self.iidW2 = tf.Variable(
                tf.constant(0.1, shape=[self.n_items + 2]), name="iidW2"
            )
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.feature_bias = self.u_bias + self.i_bias

            self.biased = tf.Variable(tf.constant(0.1), name="bias")

            self.predictions = self.score + self.feature_bias + self.biased

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))
            self.loss = losses + self.l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(
                tf.abs(tf.subtract(self.predictions, self.input_y))
            )
            self.rmse = tf.sqrt(
                tf.reduce_mean(tf.square(tf.subtract(
                    self.predictions, self.input_y)))
            )

        # Generate optimizer
        optimizer = tf.train.AdamOptimizer(
            self.learning_rate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon
        )
        train_var_list = [var for var in tf.trainable_variables()]
        gvs = optimizer.compute_gradients(self.loss, var_list=train_var_list)
        self.opt = optimizer.apply_gradients(gvs)
