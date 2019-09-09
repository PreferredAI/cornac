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


class Model:

    def __init__(self, n_users, n_items, k,
                 lambda_u, lambda_v, lr, U, V):
        self.n_users = n_users
        self.n_items = n_items
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lr = lr  # learning rate
        self.k = k  # latent dimension
        self.U_init = tf.constant(U)
        self.V_init = tf.constant(V)

        self._build_graph()

    def _build_graph(self):
        self.ratings = tf.placeholder(dtype=tf.float32, shape=[self.n_users, None], name="rating_input")
        self.C = tf.placeholder(dtype=tf.float32, shape=[self.n_users, None], name="C_input")
        self.item_ids = tf.placeholder(dtype=tf.int32)
        with tf.variable_scope("CF_Variable"):
            self.U = tf.get_variable(name='U', dtype=tf.float32, initializer=self.U_init)
            self.V = tf.get_variable(name='V', dtype=tf.float32, initializer=self.V_init)

        V_batch = tf.reshape(tf.gather(self.V, self.item_ids), shape=[-1, self.k])

        predictions = tf.matmul(self.U, V_batch, transpose_b=True)
        squared_error = tf.square(self.ratings - predictions)
        loss_1 = tf.reduce_sum(tf.multiply(self.C, squared_error))
        loss_2 = self.lambda_u * tf.nn.l2_loss(self.U) + self.lambda_v * tf.nn.l2_loss(V_batch)

        self.loss = loss_1 + loss_2
        # Generate optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)

        gvs = optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables())
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
        self.opt = optimizer.apply_gradients(capped_gvs)
