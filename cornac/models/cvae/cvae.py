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

import tensorflow.compat.v1 as tf

from ..cdl.cdl import act_functions


class Model:
    def __init__(
        self,
        n_users,
        n_items,
        input_dim,
        seed,
        n_z,
        layers,
        loss_type,
        act_fn,
        U,
        V,
        lambda_u,
        lambda_v,
        lambda_r,
        lambda_w,
        lr=0.001,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.input_dim = input_dim
        self.n_z = n_z
        self.layers = layers
        self.loss_type = loss_type
        self.act_fn = act_fn
        self.seed = seed
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_r = lambda_r
        self.lambda_w = lambda_w
        self.lr = lr
        self.U_init = tf.constant(U)
        self.V_init = tf.constant(V)

        self._build_graph()

    def _build_graph(self):
        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name="text_input")
        self.ratings = tf.placeholder(
            dtype=tf.float32, shape=[self.n_users, None], name="rating_input"
        )
        self.C = tf.placeholder(
            dtype=tf.float32, shape=[self.n_users, None], name="C_input"
        )
        self.item_ids = tf.placeholder(dtype=tf.int32, name="item_input")

        x_recon = self._vae(self.x)

        with tf.variable_scope("cf_variable"):
            self.U = tf.get_variable(
                name="U", dtype=tf.float32, initializer=self.U_init
            )
            self.V = tf.get_variable(
                name="V", dtype=tf.float32, initializer=self.V_init
            )

        real_batch_size = tf.cast(tf.shape(self.item_ids)[0], tf.int32)
        V_batch = tf.reshape(
            tf.gather(self.V, self.item_ids), shape=[real_batch_size, self.n_z]
        )

        predictions = tf.matmul(self.U, V_batch, transpose_b=True)
        squared_error = tf.square(self.ratings - predictions)

        # CF loss
        rating_loss = tf.reduce_mean(
            tf.reduce_sum(tf.multiply(self.C, squared_error), 0)
        )
        v_loss = (
            self.lambda_v
            / self.lambda_r
            * tf.reduce_mean(tf.reduce_sum(tf.square(V_batch - self.z), 1))
        )
        self.cf_loss = rating_loss + v_loss + self.lambda_u * tf.nn.l2_loss(self.U)

        # VAE loss
        if self.loss_type == "rmse":
            gen_loss = tf.reduce_mean(tf.square(tf.subtract(self.x, x_recon)))
        elif self.loss_type == "cross-entropy":
            x_recon = tf.nn.sigmoid(x_recon, name="x_recon")
            gen_loss = -tf.reduce_mean(
                tf.reduce_sum(
                    self.x * tf.log(tf.maximum(x_recon, 1e-10))
                    + (1 - self.x) * tf.log(tf.maximum(1 - x_recon, 1e-10)),
                    1,
                )
            )
        else:
            raise ValueError("Invalid loss type {}".format(self.loss_type))

        latent_loss = 0.5 * tf.reduce_mean(
            tf.reduce_sum(
                tf.square(self.z_mean)
                + tf.exp(self.z_log_sigma_sq)
                - self.z_log_sigma_sq
                - 1,
                1,
            )
        )
        self.vae_loss = gen_loss + latent_loss + self.lambda_w * self.reg_loss

        cf_op = tf.train.AdamOptimizer(self.lr, name="cf_op")
        vae_op = tf.train.AdamOptimizer(self.lr, name="vae_op")

        cf_var_list, vae_var_list = [], []

        for var in tf.trainable_variables():
            if "cf_variable" in var.name:
                cf_var_list.append(var)
            elif "vae" in var.name:
                vae_var_list.append(var)

        gvs = cf_op.compute_gradients(self.cf_loss, var_list=cf_var_list)
        capped_gvs = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gvs]
        self.cf_update = cf_op.apply_gradients(capped_gvs)

        gvs = vae_op.compute_gradients(self.vae_loss, var_list=vae_var_list)
        capped_gvs = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gvs]
        self.vae_update = vae_op.apply_gradients(capped_gvs)

    def _vae(self, X):
        act_fn = act_functions.get(self.act_fn, None)
        if act_fn is None:
            raise ValueError(
                "Invalid type of activation function {}\n"
                "Supported functions: {}".format(act_fn, act_functions.keys())
            )

        with tf.variable_scope("vae/inference"):
            rec = {
                "W1": tf.get_variable(
                    "W1",
                    [self.input_dim, self.layers[0]],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b1": tf.get_variable(
                    "b1",
                    [self.layers[0]],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
                "W2": tf.get_variable(
                    "W2",
                    [self.layers[0], self.layers[1]],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b2": tf.get_variable(
                    "b2",
                    [self.layers[1]],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
                "W_z_mean": tf.get_variable(
                    "W_z_mean",
                    [self.layers[1], self.n_z],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b_z_mean": tf.get_variable(
                    "b_z_mean",
                    [self.n_z],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
                "W_z_log_sigma": tf.get_variable(
                    "W_z_log_sigma",
                    [self.layers[1], self.n_z],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b_z_log_sigma": tf.get_variable(
                    "b_z_log_sigma",
                    [self.n_z],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
            }

        h1 = act_fn(tf.matmul(X, rec["W1"]) + rec["b1"])
        h2 = act_fn(tf.matmul(h1, rec["W2"]) + rec["b2"])

        self.z_mean = tf.matmul(h2, rec["W_z_mean"]) + rec["b_z_mean"]
        self.z_log_sigma_sq = tf.matmul(h2, rec["W_z_log_sigma"]) + rec["b_z_log_sigma"]

        eps = tf.random_normal(
            shape=tf.shape(self.z_mean),
            mean=0,
            stddev=1,
            seed=self.seed,
            dtype=tf.float32,
        )

        self.z = (
            self.z_mean + tf.sqrt(tf.maximum(tf.exp(self.z_log_sigma_sq), 1e-10)) * eps
        )

        with tf.variable_scope("vae/generation"):
            gen = {
                "W2": tf.get_variable(
                    "W2",
                    [self.n_z, self.layers[1]],
                    initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
                    dtype=tf.float32,
                ),
                "b2": tf.get_variable(
                    "b2",
                    [self.layers[1]],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
                "W1": tf.transpose(rec["W2"]),
                "b1": rec["b1"],
                "W_x": tf.transpose(rec["W1"]),
                "b_x": tf.get_variable(
                    "b_x",
                    [self.input_dim],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32,
                ),
            }

        self.reg_loss = (
            tf.nn.l2_loss(rec["W1"])
            + tf.nn.l2_loss(rec["W2"])
            + tf.nn.l2_loss(gen["W1"])
            + tf.nn.l2_loss(gen["W_x"])
        )

        h2 = act_fn(tf.matmul(self.z, gen["W2"]) + gen["b2"])
        h1 = act_fn(tf.matmul(h2, gen["W1"]) + gen["b1"])
        x_recon = tf.matmul(h1, gen["W_x"]) + gen["b_x"]

        return x_recon
