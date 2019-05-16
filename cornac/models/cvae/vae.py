"""
@author: Tran Thanh Binh

"""

import tensorflow as tf

from ..cdl.cdl import act_functions


class VAE():

    def __init__(self, input_dim, seed, n_z, layers, loss_type, activations, lambda_v,
                 lambda_r, lambda_w=2e-4, lr=0.001):

        self.input_dim = input_dim
        self.n_z = n_z
        self.layers = layers
        self.loss_type = loss_type
        self.activations = activations
        self.seed = seed
        self.lambda_v = lambda_v
        self.lambda_r = lambda_r
        self.lambda_w = lambda_w
        self.lr = lr

        self._build_graph()

    def _build_graph(self):

        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name='x')
        self.v = tf.placeholder(tf.float32, [None, self.n_z])

        x_recon = self._inference_generation(self.x)

        if self.loss_type == 'rmse':
            self.gen_loss = tf.reduce_mean(tf.square(tf.sub(self.x, x_recon)))
        elif self.loss_type == 'cross-entropy':
            x_recon = tf.nn.sigmoid(x_recon, name='x_recon')
            self.gen_loss = -tf.reduce_mean(tf.reduce_sum(self.x * tf.log(tf.maximum(x_recon, 1e-10))
                                                          + (1 - self.x) * tf.log(tf.maximum(1 - x_recon, 1e-10)), 1))

        latent_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(self.z_mean) + tf.exp(self.z_log_sigma_sq)
                                                         - self.z_log_sigma_sq - 1, 1))
        v_loss = 1.0 * self.lambda_v / self.lambda_r * tf.reduce_mean(
            tf.reduce_sum(tf.square(self.v - self.z), 1))

        self.loss = self.gen_loss + latent_loss + v_loss + self.lambda_w * self.reg_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _inference_generation(self, X):

        act_fn = act_functions.get(self.activations, None)
        if act_fn is None:
            raise ValueError('Invalid type of activation function {}\n'
                             'Supported functions: {}'.format(act_fn, act_functions.keys()))

        with tf.variable_scope("inference"):
            rec = {'W1': tf.get_variable("W1", [self.input_dim, self.layers[0]],
                                         initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                         dtype=tf.float32),
                   'b1': tf.get_variable("b1", [self.layers[0]],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W2': tf.get_variable("W2", [self.layers[0], self.layers[1]],
                                         initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                         dtype=tf.float32),
                   'b2': tf.get_variable("b2", [self.layers[1]],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W_z_mean': tf.get_variable("W_z_mean", [self.layers[1], self.n_z],
                                               initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                               dtype=tf.float32),
                   'b_z_mean': tf.get_variable("b_z_mean", [self.n_z],
                                               initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W_z_log_sigma': tf.get_variable("W_z_log_sigma", [self.layers[1], self.n_z],
                                                    initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                                    dtype=tf.float32),
                   'b_z_log_sigma': tf.get_variable("b_z_log_sigma", [self.n_z],
                                                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)}

        h1 = act_fn(tf.matmul(X, rec['W1']) + rec['b1'])
        h2 = act_fn(tf.matmul(h1, rec['W2']) + rec['b2'])

        self.z_mean = tf.matmul(h2, rec['W_z_mean']) + rec['b_z_mean']
        self.z_log_sigma_sq = tf.matmul(h2, rec['W_z_log_sigma']) + rec['b_z_log_sigma']

        eps = tf.random_normal(shape=tf.shape(self.z_mean), mean=0, stddev=1,
                               seed=self.seed, dtype=tf.float32)

        self.z = self.z_mean + tf.sqrt(tf.maximum(tf.exp(self.z_log_sigma_sq), 1e-10)) * eps

        with tf.variable_scope("generation"):
            gen = {'W2': tf.get_variable("W2", [self.n_z, self.layers[1]],
                                         initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                         dtype=tf.float32),
                   'b2': tf.get_variable("b2", [self.layers[1]],
                                         initializer=tf.constant_initializer(0.0), dtype=tf.float32),
                   'W1': tf.transpose(rec['W2']),
                   'b1': rec['b1'],
                   'W_x': tf.transpose(rec['W1']),
                   'b_x': tf.get_variable("b_x", [self.input_dim],
                                          initializer=tf.constant_initializer(0.0), dtype=tf.float32)}

        self.reg_loss = tf.nn.l2_loss(rec['W1']) + tf.nn.l2_loss(rec['W2']) + \
                        tf.nn.l2_loss(gen['W1']) + tf.nn.l2_loss(gen['W_x'])

        h2 = act_fn(tf.matmul(self.z, gen['W2']) + gen['b2'])
        h1 = act_fn(tf.matmul(h2, gen['W1']) + gen['b1'])
        x_recon = tf.matmul(h1, gen['W_x']) + gen['b_x']

        return x_recon
