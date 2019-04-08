# -*- coding: utf-8 -*-

"""
@author:Trieu Thi Ly Ly, Tran Thanh Binh
"""

import tensorflow as tf


class Model:

    def __init__(self, n_users, n_items, n_vocab, k,
                 layers, lambda_u, lambda_v, lambda_n, lambda_w,
                 lr, dropout_rate, U_init, V_init):
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
        self.U_init = tf.constant(U_init)
        self.V_init = tf.constant(V_init)

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
        sdae_value, encoded_value, loss_w = self._sdae(corrupted_text)

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
