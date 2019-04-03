# -*- coding: utf-8 -*-

"""
@author:Trieu Thi Ly Ly, Tran Thanh Binh
"""

import numpy as np
from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    tf = None


# Collaborative Deep Learning
def cdl(train_set, layer_sizes, k=50, lambda_u=0.01,
        lambda_v=0.01, lambda_w=0.01, lambda_n=0.01,
        a=1, b=0.01, corruption_rate=0.3, n_epochs=100,
        lr=0.001, dropout_rate=0.1, batch_size=100,
        vocab_size=8000, init_params=None, verbose=True):
    R = train_set.matrix.A  # rating matrix
    C = np.ones_like(R) * b
    C[R > 0] = a
    
    n_users = train_set.num_users
    n_items = train_set.num_items

    text_feature = train_set.item_text.batch_bow(np.arange(n_items))  # bag of word feature
    text_feature = (text_feature - text_feature.min()) / (text_feature.max() - text_feature.min())  # normalization

    layer_sizes = [vocab_size] + layer_sizes + [k] + layer_sizes + [vocab_size]

    # Build model
    model = Model(n_users=n_users, n_items=n_items, n_vocab=vocab_size, k=k, layers=layer_sizes, lambda_u=lambda_u,
                  lambda_v=lambda_v, lambda_w=lambda_w, lambda_n=lambda_n, lr=lr, dropout_rate=dropout_rate,
                  init_params=init_params)

    # Training model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loop = tqdm(range(n_epochs), disable=not verbose)
        for _ in loop:
            mask_corruption_np = np.random.binomial(1, 1 - corruption_rate,
                                                    (n_items, vocab_size))

            for batch_ids in train_set.item_iter(batch_size=batch_size, shuffle=True):
                feed_dict = {
                    model.mask_input: mask_corruption_np[batch_ids, :],
                    model.text_input: text_feature[batch_ids],
                    model.rating_input: R[:, batch_ids],
                    model.C_input: C[:, batch_ids],
                    model.cdl_batch: batch_ids
                }

                sess.run(model.opt1, feed_dict)  # train U, V
                _, _loss = sess.run([model.opt2, model.loss], feed_dict)  # train SDAE
                loop.set_postfix(loss=_loss)
        U_out, V_out = sess.run([model.U, model.V])

    return U_out.astype(np.float32), V_out.astype(np.float32)


class Model:

    def __init__(self, n_users, n_items, n_vocab, k,
                 layers, lambda_u, lambda_v, lambda_n, lambda_w,
                 lr, dropout_rate, init_params=None):

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
        self.init_params = {} if init_params is None else init_params

        self._build_graph()

    def _build_graph(self):
        self.mask_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_vocab], name="mask_input")
        self.text_input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_vocab], name="text_input")
        self.rating_input = tf.placeholder(dtype=tf.float32, shape=[self.n_users, None], name="rating_input")
        self.C_input = tf.placeholder(dtype=tf.float32, shape=[self.n_users, None], name="C_input")

        with tf.variable_scope("CDL_Variable"):
            U_init = None if 'U' not in self.init_params else tf.constant_initializer(self.init_params['U'])
            V_init = None if 'V' not in self.init_params else tf.constant_initializer(self.init_params['V'])
            self.U = tf.get_variable(name='U', shape=[self.n_users, self.k], dtype=tf.float32, initializer=U_init)
            self.V = tf.get_variable(name='V', shape=[self.n_items, self.k], dtype=tf.float32, initializer=V_init)

        self.cdl_batch = tf.placeholder(dtype=tf.int32)
        real_batch_size = tf.cast(tf.shape(self.text_input)[0], tf.int32)
        V_batch = tf.reshape(tf.gather(self.V, self.cdl_batch), shape=[real_batch_size, self.k])

        corrupted_text = tf.multiply(self.text_input, self.mask_input)
        sdae_value, encoded_value, loss_w = self._sdae(corrupted_text)

        loss_1 = self.lambda_u * tf.nn.l2_loss(self.U) + self.lambda_w * loss_w
        loss_2 = self.lambda_v * tf.nn.l2_loss(V_batch - encoded_value)
        loss_3 = self.lambda_n * tf.nn.l2_loss(sdae_value - self.text_input)
        loss_4 = tf.reduce_sum(tf.multiply(self.C_input,
                                           tf.square(self.rating_input - tf.matmul(self.U, V_batch, transpose_b=True))))

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
