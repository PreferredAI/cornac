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

import tensorflow as tf


def conv_layer(input, num_input_channels,
               filter_height, filter_width,
               num_filters, seed=None, use_pooling=True):
    shape = [filter_height, filter_width, num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05, seed=seed))
    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
    layer = tf.nn.conv2d(input=input, filter=weights,
                         strides=[1, 1, 1, 1], padding="VALID")
    layer = layer + biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, input.shape[1] - filter_height + 1, 1, 1],
                               strides=[1, 1, 1, 1], padding="VALID")
    layer = tf.nn.relu(layer)
    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_feature = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_feature])
    return layer_flat, num_feature


def fc_layer(input, num_input, num_output, seed=None):
    weights = tf.Variable(tf.truncated_normal([num_input, num_output], stddev=0.05, seed=seed))
    biases = tf.Variable(tf.constant(0.05, shape=[num_output]))
    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.tanh(layer)
    return layer


class CNN_module():

    def __init__(self, output_dimension, dropout_rate,
                 emb_dim, max_len, nb_filters, seed,
                 init_W, learning_rate=0.001):
        self.drop_rate = dropout_rate
        self.max_len = max_len
        self.seed = seed
        self.learning_rate = learning_rate
        self.init_W = tf.constant(init_W)
        self.output_dimension = output_dimension
        self.emb_dim = emb_dim
        self.nb_filters = nb_filters
        self.filter_lengths = [3, 4, 5]
        self.vanila_dimension = 200

        self._build_graph()

    def _build_graph(self):
        # create Graph
        self.model_input = tf.placeholder(dtype=tf.int32, shape=(None, self.max_len))
        self.v = tf.placeholder(dtype=tf.float32, shape=(None, self.output_dimension))
        self.sample_weight = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.embedding_weight = tf.Variable(initial_value=self.init_W)

        self.seq_emb = tf.nn.embedding_lookup(self.embedding_weight, self.model_input)
        self.reshape = tf.reshape(self.seq_emb, [-1, self.max_len, self.emb_dim, 1])
        self.convs = []

        # Convolutional layer
        for i in self.filter_lengths:
            convolutional_layer, weights = conv_layer(input=self.reshape, num_input_channels=1,
                                                      filter_height=i, filter_width=self.emb_dim,
                                                      num_filters=self.nb_filters, use_pooling=True)

            flat_layer, _ = flatten_layer(convolutional_layer)
            self.convs.append(flat_layer)

        self.model_output = tf.concat(self.convs, axis=-1)
        # Fully-connected layers
        self.model_output = fc_layer(input=self.model_output, num_input=self.model_input.get_shape()[1].value,
                                     num_output=self.vanila_dimension)
        # Dropout layer
        self.model_output = tf.nn.dropout(self.model_output, self.drop_rate)
        # Output layer
        self.model_output = fc_layer(input=self.model_output, num_input=self.vanila_dimension,
                                     num_output=self.output_dimension)
        # Weighted MEA loss function
        self.mean_square_loss = tf.losses.mean_squared_error(labels=self.v, predictions=self.model_output,
                                                             reduction=tf.losses.Reduction.NONE)
        self.weighted_loss = tf.reduce_sum(
            tf.reduce_sum(self.mean_square_loss, axis=1, keepdims=True) * self.sample_weight)
        # RMSPro optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.weighted_loss)
