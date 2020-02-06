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
import scipy.sparse as sp
import scipy as sc
import tensorflow as tf


class PCRL_:
    def __init__(
        self,
        train_set,
        k=100,
        z_dims=[300],
        n_epoch=300,
        batch_size=300,
        learning_rate=0.001,
        B=1,
        w_determinist=True,
        init_params=None,
    ):

        self.train_set = train_set
        self.cf_data = sp.csc_matrix(
            self.train_set.matrix
        )  # user-item interaction (CF data)
        self.aux_data = self.train_set.item_graph.matrix[
            : self.train_set.num_items, : self.train_set.num_items
        ]  # item auxiliary information (items'context in the original paper)
        self.k = k  # the number of user and item latent factors
        self.z_dims = (
            z_dims
        )  # the dimension of the second hidden layer (we consider a 2-layers PCRL)
        self.c_dim = self.aux_data.shape[
            1
        ]  # the dimension of the auxiliary information matrix
        self.n_epoch = n_epoch  # the number of traning epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.B = B  # Shape augmentation parameter
        self.init_params = init_params  # initial parameters
        # Additional parameters
        self.aa = 0.3
        self.bb = 0.3
        self.Ls = sp.csc_matrix(
            (self.aux_data.shape[0], self.k)
        )  # Variational Shape parameters of the item factors (Beta in the paper)
        self.Lr = sp.csc_matrix(
            (self.aux_data.shape[0], self.k)
        )  # Variational Rate parameters  of the item factors (Beta in the paper)
        self.Gs = (
            None
        )  # Variational Shapre parameters of the user factors (Theta in the paper)
        self.Gr = (
            None
        )  # Variational Rate parameters of the user factors (Theta in the paper)
        self.L = len(z_dims)  # The number of deterministic hidden layers "z"
        self.w_determinist = (
            w_determinist
        )  # If true then deterministic wheights are used for the generator network
        self.sess = tf.Session()  # Tensorflow session
        # Inference netwok parameters
        self.inference_params = []
        self.inference_params.append(
            tf.Variable(self.glorot_init([self.c_dim, self.z_dims[self.L - 1]]))
        )
        for l in range(self.L - 2, -1, -1):
            self.inference_params.append(
                tf.Variable(self.glorot_init([self.z_dims[l + 1], self.z_dims[l]]))
            )
        self.inference_params.append(
            tf.Variable(self.glorot_init([self.z_dims[0], self.k]))
        )
        self.inference_params.append(
            tf.Variable(self.glorot_init([self.z_dims[0], self.k]))
        )
        # generator newtork parameters
        self.generator_params = []
        self.generator_params.append(
            tf.Variable(self.glorot_init([self.k, self.z_dims[0]]))
        )
        for l in range(1, self.L):
            self.generator_params.append(
                tf.Variable(self.glorot_init([self.z_dims[l - 1], self.z_dims[l]]))
            )
        self.generator_params.append(
            tf.Variable(self.glorot_init([self.z_dims[self.L - 1], self.c_dim]))
        )

    def glorot_init(self, shape):
        return tf.random_normal(shape=shape, stddev=1.0 / tf.sqrt(shape[0] / 2.0))

    # some until function to compute the loss
    # Log density of Ga(alpha, beta)
    def log_q(self, z, alpha, beta):
        return (
            (alpha - 1) * tf.log(z) - beta * z + alpha * tf.log(beta) - tf.lgamma(alpha)
        )

        # Log density of the standard normal N(0, 1)

    def log_t(self, epsilon):
        return -0.5 * tf.log(2 * np.pi) - 0.5 * epsilon ** 2

    # Marsaglia and Tsang transformation
    def G(self, epsilon, alpha, beta):
        return (
            (alpha - 1.0 / 3.0) * (1 + epsilon / tf.sqrt(9.0 * alpha - 3.0)) ** 3 / beta
        )

    # derivative of h
    def dG(self, epsilon, alpha, beta):
        return (
            (alpha - 1.0 / 3)
            * (3.0 / tf.sqrt(9.0 * alpha - 3.0))
            * (1.0 + epsilon / tf.sqrt(9.0 * alpha - 3.0)) ** 2
        ) / beta

    # Log density of the proposal distribution r(z) = t(epsilon) * |dG/depsilon|^{-1}
    def log_r(self, epsilon, alpha, beta):
        return -tf.log(self.dG(epsilon, alpha, beta)) + self.log_t(epsilon)

    # Inverse transformation
    def G_inv(self, z, alpha, beta):
        return tf.sqrt(9.0 * alpha - 3.0) * (
            (beta * z / (alpha - 1.0 / 3.0)) ** (1.0 / 3.0) - 1.0
        )

    # Sample from the marginal of the accepted epsilon's, epsilon ~ pi(epsilon)
    def sample_pi(self, alpha, beta):
        Gam = tf.random_gamma([1], alpha=alpha, beta=beta, name="Gam", seed=None)[0]
        return self.G_inv(Gam, alpha, beta)

    # shape augmentation
    def shape_augmentation(self, alpha, B):
        UUU = 1.0
        for i in range(int(B)):
            UUU = UUU * tf.pow(
                tf.random_uniform(tf.shape(alpha), seed=None), 1.0 / (alpha + i)
            )
        return UUU

    # Collaborative filtering part of pcrl (Poisson Factorization)
    def pf_(self, X, k, max_iter=1, init_params=None):

        # data preparation
        X = sp.csc_matrix(X, dtype=np.float64)
        M = X.copy()
        M.data = np.full(len(M.data), 1)
        n = X.shape[0]
        d = X.shape[1]
        eps = 0.000000001

        # Hyper-parameters setting
        a = 0.3

        # Parameter initialization

        # shape gamma_uk matrix (dgCMatrix)
        if init_params["G_s"] is None:
            G_s = np.random.gamma(50, scale=0.3 / 50, size=n * k).reshape(n, k)
        else:
            G_s = init_params["G_s"]
        G_s = sp.csc_matrix(G_s, dtype=np.float64)

        ## rate gamma_uk matrix (dgCMatrix)
        if init_params["G_r"] is None:
            G_r = np.random.gamma(50, scale=0.3 / 50, size=n * k).reshape(n, k)
        else:
            G_r = init_params["G_r"]
        G_r = sp.csc_matrix(G_r, dtype=np.float64)

        # shape lamda_ik matrix (dgCMatrix)
        if init_params["L_s"] is None:
            L_s = np.random.gamma(50, scale=0.3 / 50, size=d * k).reshape(d, k)
        else:
            L_s = init_params["L_s"]
        L_s = sp.csc_matrix(L_s, dtype=np.float64)

        ## rate lamda_ik matrix (dgCMatrix)
        if init_params["L_r"] is None:
            L_r = np.random.gamma(50, scale=0.3 / 50, size=d * k).reshape(d, k)
        else:
            L_r = init_params["L_r"]
        L_r = sp.csc_matrix(L_r, dtype=np.float64)

        # need to be computed only once as Lr and Ls don't change here
        logL_r = L_r.copy()
        logL_r.data = np.log(logL_r.data)
        digaL_s = L_s.copy()
        digaL_s.data = sc.special.digamma(digaL_s.data)

        Lb = digaL_s - logL_r
        Lb.data = np.exp(Lb.data)

        del logL_r
        del digaL_s

        Lb = Lb.todense()

        # Learning
        for iter_ in range(1, max_iter + 1):
            ## Update multinomiale parameter, no need to store phi only compute the sufficient stats
            logG_r = G_r.copy()
            logG_r.data = np.log(logG_r.data)
            digaG_s = G_s.copy()
            digaG_s.data = sc.special.digamma(digaG_s.data)

            Lt = digaG_s - logG_r
            Lt.data = np.exp(Lt.data)

            del logG_r
            del digaG_s

            Lt = Lt.todense()

            ## Update user related parameters
            G_s = a + np.multiply(Lt, ((X / (Lt * Lb.T + eps)) * Lb))
            G_r = np.repeat(np.sum(L_s / L_r, 0), n, axis=0) + a

            G_s = sp.csc_matrix(G_s)
            G_r = sp.csc_matrix(G_r)

        Tk = np.repeat(np.sum(G_s / G_r, 0), self.batch_size, axis=0)
        Zik = np.multiply(Lb, ((X.T / (Lb * Lt.T + eps)) * Lt))
        # End of learning

        res = {
            "Z": G_s / G_r,
            "W": L_s / L_r,
            "G_s": G_s,
            "G_r": G_r,
            "L_s": L_s,
            "L_r": L_r,
            "Lt": Lt,
            "Lb": Lb,
            "Zik": np.array(Zik, dtype="float32"),
            "Tk": np.array(Tk, dtype="float32"),
        }

        return res

    # The inference network (or encoder)
    def inference_net(self, C, reuse=None):
        # input
        # h = tf.nn.relu(tf.sparse_matmul(C, self.inference_params[0], a_is_sparse=True))
        h = tf.nn.relu(tf.matmul(C, self.inference_params[0]))
        # intermediate hidden layer
        for l in range(1, self.L):
            # h = tf.nn.relu(tf.sparse_matmul(h, self.inference_params[l], a_is_sparse=True))
            h = tf.nn.relu(tf.matmul(h, self.inference_params[l]))

        # output
        # beta = tf.nn.softplus(tf.matmul(h, self.inference_params[self.L], a_is_sparse=True)) + 0.3
        # alpha = tf.nn.softplus(tf.matmul(h, self.inference_params[self.L + 1], a_is_sparse=True)) + 0.3
        beta = tf.nn.softplus(tf.matmul(h, self.inference_params[self.L])) + 0.3
        alpha = tf.nn.softplus(tf.matmul(h, self.inference_params[self.L + 1])) + 0.3

        return alpha, beta

    # The generative network (or decoder)
    def generative_net(self, Z, reuse=None):

        # with tf.variable_scope("generative",reuse=reuse):
        if self.w_determinist:
            h2 = tf.nn.relu(tf.matmul(Z, self.generator_params[0]))
            for l in range(1, self.L):
                # h2 = tf.nn.relu(tf.sparse_matmul(h2, self.generator_params[l], a_is_sparse=True))
                h2 = tf.nn.relu(tf.matmul(h2, self.generator_params[l]))
            # d_x = tf.nn.sigmoid(tf.matmul(h2, self.generator_params[self.L], a_is_sparse=True))
            d_x = tf.nn.sigmoid(tf.matmul(h2, self.generator_params[self.L]))
        else:
            e = tf.random_normal(
                tf.shape(self.generator_params[0]),
                dtype=tf.float32,
                mean=0.0,
                stddev=1.0,
                name="epsilon",
            )
            h2 = tf.nn.relu(tf.matmul(Z, self.generator_params[0] + 0.01 * e))
            for l in range(1, self.L):
                e = tf.random_normal(
                    tf.shape(self.generator_params[l]),
                    dtype=tf.float32,
                    mean=0.0,
                    stddev=1.0,
                    name="epsilon",
                )
                # h2 = tf.nn.relu(tf.sparse_matmul(h2, self.generator_params[l] + 0.01 * e, a_is_sparse=True))
                h2 = tf.nn.relu(tf.matmul(h2, self.generator_params[l] + 0.01 * e))
            e = tf.random_normal(
                tf.shape(self.generator_params[self.L]),
                dtype=tf.float32,
                mean=0.0,
                stddev=1.0,
                name="epsilon",
            )
            # d_x = tf.nn.sigmoid(tf.matmul(h2, self.generator_params[self.L] + 0.01 * e, a_is_sparse=True))
            d_x = tf.nn.sigmoid(tf.matmul(h2, self.generator_params[self.L] + 0.01 * e))
        return d_x

        # The loss function

    def loss(self, C, X_g, X_, alpha, beta, z, E, Zik, Tk):

        const_term = C * tf.log(1e-10 + X_) - X_
        const_term = tf.reduce_sum(const_term, 1)

        loss1 = C * tf.log(1e-10 + X_g) - X_g
        loss1 = tf.reduce_sum(loss1, 1)

        loss2 = self.log_q(z, alpha + self.B, beta)
        loss2 = const_term * tf.reduce_sum(loss2, 1)

        # loss3 = -log_r(E, alpha,beta)
        loss3 = -self.log_r(E, alpha + self.B, beta)
        loss3 = const_term * tf.reduce_sum(loss3, 1)

        # The sum of KL terms of all generator's wheights (up to constant terms)
        kl_w = 0.0
        if not self.w_determinist:
            for l in range(0, self.L + 1):
                kl_w += tf.reduce_sum(
                    -0.5 * tf.reduce_sum(tf.square(self.generator_params[l]), 1)
                )

        # KL Divergence term
        kl_term = (
            (alpha - self.aa - Zik) * tf.digamma(alpha)
            - tf.lgamma(alpha)
            + (self.aa + Zik) * tf.log(beta)
            + alpha * (Tk + self.bb - beta) / beta
        )
        kl_term = -tf.reduce_sum(kl_term, 1)
        return (
            -tf.reduce_mean(loss1 + loss2 + loss3 + kl_term)
            + kl_w / self.aux_data.shape[0]
        )

    # fitting PCRL to observed data
    def learn(self):
        # placeholders
        C = tf.placeholder(tf.float32, shape=[None, self.c_dim], name="C")
        X_ = tf.placeholder(tf.float32, shape=[None, self.c_dim], name="X_")
        Zik = tf.placeholder(tf.float32, shape=[None, self.k], name="Zik")
        Tk = tf.placeholder(tf.float32, shape=[None, self.k], name="Tk")
        E = tf.placeholder(
            tf.float32, shape=[None, self.k], name="E"
        )  # matrix of accepted samples epsilon, epsilon ~ pi(epsilon)

        # Sample Gamma variables using the Reparameterized Acceptance-Rejection approach
        alpha, beta = self.inference_net(C)
        E_ = self.sample_pi(alpha + self.B, beta)
        z_tld = self.G(E, alpha + self.B, beta)
        # getting the right z, this step is necessary when alpha<1.0
        U_ = self.shape_augmentation(alpha, self.B)
        z = U_ * z_tld

        # Generating
        X_g = self.generative_net(z)

        # preparing optimization
        loss = self.loss(C, X_g, X_, alpha, beta, z_tld, E, Zik, Tk)
        # train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        train = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(
            loss
        )

        # Initialization
        init = tf.global_variables_initializer()

        # Learning
        self.sess.run(init)

        # Train the collaborative part
        resPF = self.pf_(
            self.cf_data, k=self.k, max_iter=1, init_params=self.init_params
        )

        for epoch in range(self.n_epoch):
            for idx in self.train_set.item_iter(self.batch_size, shuffle=False):
                batch_C = self.aux_data[idx].A
                EE = self.sess.run(E_, feed_dict={C: batch_C})
                z_c = self.sess.run(X_g, feed_dict={C: batch_C, E: EE})
                feed_dict = {
                    C: batch_C,
                    X_: z_c,
                    E: EE,
                    Zik: resPF["Zik"][idx],
                    Tk: resPF["Tk"][0 : len(idx)],
                }
                _, l = self.sess.run([train, loss], feed_dict=feed_dict)
                del (EE, z_c)
            for idx in self.train_set.item_iter(2 * self.batch_size, shuffle=False):
                batch_C = self.aux_data[idx].A
                self.Ls[idx], self.Lr[idx] = self.sess.run(
                    [alpha, beta], feed_dict={C: batch_C}
                )
            print("epoch %i, Train Loss: %f" % (epoch, l))
            resPF = self.pf_(
                self.cf_data,
                k=self.k,
                max_iter=1,
                init_params={
                    "G_s": resPF["G_s"],
                    "G_r": resPF["G_r"],
                    "L_s": self.Ls,
                    "L_r": self.Lr,
                },
            )
        self.Gs = resPF["G_s"]
        self.Gr = resPF["G_r"]
        print("learning done successfully")
        # End of traning

        return self
