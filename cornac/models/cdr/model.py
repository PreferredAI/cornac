# -*- coding: utf-8 -*-

"""
@author:Tran Thanh Binh

Collaborative Deep Ranking model
"""

import tensorflow as tf


class Model:
    def __init__(self, n_users, n_items, n_vocab, k,
                 layers, lambda_u, lambda_v, lambda_n, lambda_w,
                 lr, dropout_rate, U, V):

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
        self._build_graph(U_init=U, V_init=V)

    def _build_graph(self, U_init, V_init):
        self.mask_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_vocab], name="mask_input")
        self.text_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_vocab], name="text_input")

        with tf.variable_scope("CDR_Variable"):
            self.U = tf.get_variable(name='U', shape=[self.n_users, self.k], dtype=tf.float32,
                                     initializer=tf.constant_initializer(U_init))

            self.V = tf.get_variable(name='V', shape=[self.n_items, self.k], dtype=tf.float32,
                                     initializer=tf.constant_initializer(V_init))

        self.batch_u = tf.placeholder(dtype=tf.int32)
        self.batch_i = tf.placeholder(dtype=tf.int32)
        self.batch_j = tf.placeholder(dtype=tf.int32)

        real_batch_size = tf.cast(tf.shape(self.batch_u)[0], tf.int32)

        U_batch = tf.reshape(tf.gather(self.U, self.batch_u), shape=[real_batch_size, self.k])
        I_batch = tf.reshape(tf.gather(self.V, self.batch_i), shape=[real_batch_size, self.k])
        J_batch = tf.reshape(tf.gather(self.V, self.batch_j), shape=[real_batch_size, self.k])

        corrupted_text = tf.multiply(self.text_input, self.mask_input)

        sdae_value, encoded_value, loss_w = self._sdae(corrupted_text)

        loss_1 = self.lambda_u * tf.nn.l2_loss(U_batch) \
                 + self.lambda_w * loss_w
        loss_2 = self.lambda_v * tf.nn.l2_loss(I_batch - encoded_value)

        loss_3 = self.lambda_n * tf.nn.l2_loss(sdae_value - self.text_input)

        ui_score = tf.reduce_sum(tf.multiply(U_batch, I_batch), axis=1)
        uj_score = tf.reduce_sum(tf.multiply(U_batch, J_batch), axis=1)

        # loss function for optimizing ranking
        loss_4 = tf.nn.l2_loss(1 - (ui_score - uj_score))

        self.loss = loss_1 + loss_2 + loss_3 + loss_4

        # Generate optimizer
        optimizer1 = tf.train.AdamOptimizer(self.lr)
        optimizer2 = tf.train.AdamOptimizer(self.lr)

        train_var_list1 = [var for var in tf.trainable_variables() if "CDR_Variable" in var.name]
        train_var_list2 = [var for var in tf.trainable_variables() if "SDAE_Variable" in var.name]

        gvs = optimizer1.compute_gradients(self.loss, var_list=train_var_list1)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.opt1 = optimizer1.apply_gradients(capped_gvs)

        gvs = optimizer2.compute_gradients(self.loss, var_list=train_var_list2)
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.opt2 = optimizer2.apply_gradients(capped_gvs)

    # Stacked Denoising Autoencoder
    def _sdae(self, X_corrupted):
        L = len(self.layers)
        weights, biases = dict(), dict()
        with tf.variable_scope("SDAE_Variable"):
            for i in range(L - 1):
                weights[i] = tf.get_variable(name='W_{}'.format(i), dtype=tf.float32,
                                             shape=(self.layers[i], self.layers[i + 1]))
                biases[i] = tf.get_variable(name='b_{}'.format(i), dtype=tf.float32,
                                            shape=(self.layers[i + 1]),
                                            initializer=tf.zeros_initializer())
        for i in range(L - 1):
            # The encoder
            if i <= int(L / 2) - 1:
                if i == 0:
                    # The first layer
                    h_value = tf.nn.relu(tf.add(tf.matmul(X_corrupted, weights[i]), biases[i]))
                else:
                    h_value = tf.nn.relu(tf.add(tf.matmul(h_value, weights[i]), biases[i]))
            # The decoder
            elif i > int(L / 2) - 1:
                h_value = tf.nn.relu(tf.add(tf.matmul(h_value, weights[i]), biases[i]))
            # The dropout for all the layers except the final one
            if i < (L - 2):
                h_value = tf.nn.dropout(h_value, rate=self.dropout_rate)
            # The encoder output value
            if i == int(L / 2) - 1:
                encoded_value = h_value

        sdae_value = h_value

        # Generate loss function
        loss_w = tf.constant(0, dtype=tf.float32)
        for i in range(L - 1):
            loss_w = tf.add(loss_w, tf.nn.l2_loss(weights[i]))

        return sdae_value, encoded_value, loss_w
