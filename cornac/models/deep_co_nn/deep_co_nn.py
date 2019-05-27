"""
@author: Tran Thanh Binh

"""

import tensorflow as tf
import numpy as np


def conv_layer(input, num_input_channels,
               filter_height, filter_width,
               num_filters, seed=None, use_pooling=True):
    shape = [filter_height, filter_width, num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05, seed=seed, name="W"))
    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]), name="b")
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


class Module():

    def __init__(self, output_dimension, dropout_rate,
                 emb_dim, max_len, nb_filters, seed,
                 Wu_embed, Wi_embed, learning_rate=0.001):

        self.drop_rate = dropout_rate
        self.max_len = max_len
        self.seed = seed
        self.learning_rate = learning_rate
        self.Wu_embed = tf.constant(Wu_embed)
        self.Wi_embed = tf.constant(Wi_embed)
        self.output_dimension = output_dimension
        self.emb_dim = emb_dim
        self.nb_filters = nb_filters
        self.filter_lengths = [3, 4, 5]
        self.vanila_dimension = 200

        self._build_graph()

    def _build_graph(self):
        # create Graph

        self.u_feature = self._build_branch("user")
        self.i_feature = self._build_branch("item")
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.output_dimension), name="y")
        self.drop_out = tf.placeholder(tf.float32, name="dropout_placeholder")
        self.z = tf.concat(values=[self.u_feature, self.i_feature], axis=1)



        # RMSPro optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.weighted_loss)

    def _build_branch(self, name):

        if name == "user":
            W_embed = self.Wu_embed
        elif name == "item":
            W_embed = self.Wi_embed
        else:
            raise ValueError('Unknown name {}'.format(name))

        with tf.name_scope(name):
            model_input = tf.placeholder(dtype=tf.int32, shape=(None, self.max_len), name="input")
            embedding_weight = tf.Variable(initial_value=W_embed, name="embedding")
            seq_emb = tf.nn.embedding_lookup(embedding_weight, model_input)
            reshape = tf.reshape(seq_emb, [-1, self.max_len, self.emb_dim, 1])
            convs = []
            # Convolutional layer
            for i in self.filter_lengths:
                convolutional_layer, weights = conv_layer(input=reshape, num_input_channels=1,
                                                          filter_height=i, filter_width=self.emb_dim,
                                                          num_filters=self.nb_filters, use_pooling=True)
                flat_layer, _ = flatten_layer(convolutional_layer)
                convs.append(flat_layer)

            model_output = tf.concat(convs, axis=-1)
            model_output = fc_layer(input=model_output, num_input=model_input.get_shape()[1].value,
                                    num_output=self.vanila_dimension)

            return model_output
