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

from ...utils import get_rng
from ...utils.init_utils import xavier_uniform

act_functions = {
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
    'elu': tf.nn.elu,
    'relu': tf.nn.relu,
    'relu6': tf.nn.relu6,
    'leaky_relu': tf.nn.leaky_relu,
    'identity': tf.identity
}


# Stacked Denoising Autoencoder
def sdae(X_corrupted, layers, dropout_rate=0.0, act_fn='relu', seed=None, name="SDAE"):
    fn = act_functions.get(act_fn, None)
    if fn is None:
        raise ValueError('Invalid type of activation function {}\n'
                         'Supported functions: {}'.format(act_fn, act_functions.keys()))

    # Weight initialization
    L = len(layers)
    rng = get_rng(seed)
    weights, biases = [], []
    with tf.variable_scope(name):
        for i in range(L - 1):
            w = xavier_uniform((layers[i], layers[i + 1]), random_state=rng)
            weights.append(tf.get_variable(name='W_{}'.format(i), dtype=tf.float32,
                                           initializer=tf.constant(w)))
            biases.append(tf.get_variable(name='b_{}'.format(i), dtype=tf.float32,
                                          shape=(layers[i + 1]),
                                          initializer=tf.zeros_initializer()))

    # Construct the auto-encoder
    h_value = X_corrupted
    for i in range(L - 1):
        # The encoder
        if i <= int(L / 2) - 1:
            h_value = fn(tf.add(tf.matmul(h_value, weights[i]), biases[i]))
        # The decoder
        elif i > int(L / 2) - 1:
            h_value = fn(tf.add(tf.matmul(h_value, weights[i]), biases[i]))
        # The dropout for all the layers except the final one
        if i < (L - 2):
            h_value = tf.nn.dropout(h_value, rate=dropout_rate)
        # The encoder output value
        if i == int(L / 2) - 1:
            encoded_value = h_value

    sdae_value = h_value

    # L2 loss
    loss_w = tf.constant(0.0)
    for i in range(L - 1):
        loss_w = tf.add(loss_w, tf.nn.l2_loss(weights[i]) + tf.nn.l2_loss(biases[i]))

    return sdae_value, encoded_value, loss_w


class Model:

    def __init__(self, n_users, n_items, n_vocab, k,
                 layers, lambda_u, lambda_v, lambda_n, lambda_w,
                 lr, dropout_rate, U, V, act_fn, seed):
        self.n_vocab = n_vocab
        self.n_users = n_users
        self.n_items = n_items
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_n = lambda_n
        self.lambda_w = lambda_w
        self.layers = layers
        self.lr = lr  # learning rate
        self.k = k  # latent dimension
        self.dropout_rate = dropout_rate
        self.U_init = tf.constant(U)
        self.V_init = tf.constant(V)
        self.act_fn = act_fn
        self.seed = seed

        self._build_graph()

    def _build_graph(self):
        self.text_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.n_vocab], name="mask_input")
        self.text_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_vocab], name="text_input")
        self.ratings = tf.placeholder(dtype=tf.float32, shape=[self.n_users, None], name="rating_input")
        self.C = tf.placeholder(dtype=tf.float32, shape=[self.n_users, None], name="C_input")

        with tf.variable_scope("CDL_Variable"):
            self.U = tf.get_variable(name='U', dtype=tf.float32, initializer=self.U_init)
            self.V = tf.get_variable(name='V', dtype=tf.float32, initializer=self.V_init)

        self.item_ids = tf.placeholder(dtype=tf.int32)
        real_batch_size = tf.cast(tf.shape(self.text_input)[0], tf.int32)
        V_batch = tf.reshape(tf.gather(self.V, self.item_ids), shape=[real_batch_size, self.k])

        corrupted_text = tf.multiply(self.text_input, self.text_mask)
        sdae_value, encoded_value, loss_w = sdae(X_corrupted=corrupted_text, layers=self.layers,
                                                 dropout_rate=self.dropout_rate, act_fn=self.act_fn,
                                                 seed=self.seed, name='SDAE_Variable')

        loss_1 = self.lambda_u * tf.nn.l2_loss(self.U) + self.lambda_w * loss_w
        loss_2 = self.lambda_v * tf.nn.l2_loss(V_batch - encoded_value)
        loss_3 = self.lambda_n * tf.nn.l2_loss(sdae_value - self.text_input)

        predictions = tf.matmul(self.U, V_batch, transpose_b=True)
        squared_error = tf.square(self.ratings - predictions)
        loss_4 = tf.reduce_sum(tf.multiply(self.C, squared_error))

        self.loss = loss_1 + loss_2 + loss_3 + loss_4

        # Generate optimizer
        optimizer1 = tf.train.AdamOptimizer(self.lr)
        optimizer2 = tf.train.AdamOptimizer(self.lr)

        train_var_list1, train_var_list2 = [], []

        for var in tf.trainable_variables():
            if "CDL_Variable" in var.name:
                train_var_list1.append(var)
            elif "SDAE_Variable" in var.name:
                train_var_list2.append(var)

        gvs = optimizer1.compute_gradients(self.loss, var_list=train_var_list1)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.opt1 = optimizer1.apply_gradients(capped_gvs)

        gvs = optimizer2.compute_gradients(self.loss, var_list=train_var_list2)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.opt2 = optimizer2.apply_gradients(capped_gvs)
