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
"""Collaborative Deep Ranking model"""

import tensorflow as tf

from ..cdl.cdl import sdae


class Model:
    def __init__(
        self,
        n_users,
        n_items,
        n_vocab,
        k,
        layers,
        lambda_u,
        lambda_v,
        lambda_n,
        lambda_w,
        lr,
        dropout_rate,
        U,
        V,
        act_fn,
        seed,
    ):
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
        self.mask_input = tf.placeholder(
            dtype=tf.float32, shape=[None, self.n_vocab], name="mask_input"
        )
        self.text_input = tf.placeholder(
            dtype=tf.float32, shape=[None, self.n_vocab], name="text_input"
        )

        with tf.variable_scope("CDR_Variable"):
            self.U = tf.get_variable(
                name="U", dtype=tf.float32, initializer=self.U_init
            )
            self.V = tf.get_variable(
                name="V", dtype=tf.float32, initializer=self.V_init
            )

        self.batch_u = tf.placeholder(dtype=tf.int32)
        self.batch_i = tf.placeholder(dtype=tf.int32)
        self.batch_j = tf.placeholder(dtype=tf.int32)

        real_batch_size = tf.cast(tf.shape(self.batch_u)[0], tf.int32)

        U_batch = tf.reshape(
            tf.gather(self.U, self.batch_u), shape=[real_batch_size, self.k]
        )
        I_batch = tf.reshape(
            tf.gather(self.V, self.batch_i), shape=[real_batch_size, self.k]
        )
        J_batch = tf.reshape(
            tf.gather(self.V, self.batch_j), shape=[real_batch_size, self.k]
        )

        corrupted_text = tf.multiply(self.text_input, self.mask_input)
        sdae_value, encoded_value, loss_w = sdae(
            X_corrupted=corrupted_text,
            layers=self.layers,
            dropout_rate=self.dropout_rate,
            act_fn=self.act_fn,
            seed=self.seed,
            name="SDAE_Variable",
        )

        loss_1 = self.lambda_u * tf.nn.l2_loss(U_batch) + self.lambda_w * loss_w
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

        train_var_list1 = [
            var for var in tf.trainable_variables() if "CDR_Variable" in var.name
        ]
        train_var_list2 = [
            var for var in tf.trainable_variables() if "SDAE_Variable" in var.name
        ]

        gvs = optimizer1.compute_gradients(self.loss, var_list=train_var_list1)
        capped_gvs = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gvs]
        self.opt1 = optimizer1.apply_gradients(capped_gvs)

        gvs = optimizer2.compute_gradients(self.loss, var_list=train_var_list2)
        capped_gvs = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gvs]
        self.opt2 = optimizer2.apply_gradients(capped_gvs)
