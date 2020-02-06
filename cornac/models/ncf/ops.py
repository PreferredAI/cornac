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


act_functions = {
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh,
    "elu": tf.nn.elu,
    "selu": tf.nn.selu,
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "leaky_relu": tf.nn.leaky_relu,
}


def loss_fn(labels, logits):
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    reg_loss = tf.losses.get_regularization_loss()
    return cross_entropy + reg_loss


def train_fn(loss, learning_rate, learner):
    if learner.lower() == "adagrad":
        opt = tf.train.AdagradOptimizer(learning_rate=learning_rate, name="optimizer")
    elif learner.lower() == "rmsprop":
        opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name="optimizer")
    elif learner.lower() == "adam":
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer")
    else:
        opt = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate, name="optimizer"
        )

    return opt.minimize(loss)


def emb(
    uid, iid, num_users, num_items, emb_size, reg_user, reg_item, seed=None, scope="emb"
):
    with tf.variable_scope(scope):
        user_emb = tf.get_variable(
            "user_emb",
            shape=[num_users, emb_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
            regularizer=tf.contrib.layers.l2_regularizer(scale=reg_user),
        )
        item_emb = tf.get_variable(
            "item_emb",
            shape=[num_items, emb_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=0.01, seed=seed),
            regularizer=tf.contrib.layers.l2_regularizer(scale=reg_item),
        )

    return tf.nn.embedding_lookup(user_emb, uid), tf.nn.embedding_lookup(item_emb, iid)


def gmf(uid, iid, num_users, num_items, emb_size, reg_user, reg_item, seed=None):
    with tf.variable_scope("GMF") as scope:
        user_emb, item_emb = emb(
            uid=uid,
            iid=iid,
            num_users=num_users,
            num_items=num_items,
            emb_size=emb_size,
            reg_user=reg_user,
            reg_item=reg_item,
            seed=seed,
            scope=scope,
        )
        return tf.multiply(user_emb, item_emb)


def mlp(uid, iid, num_users, num_items, layers, reg_layers, act_fn, seed=None):
    with tf.variable_scope("MLP") as scope:
        user_emb, item_emb = emb(
            uid=uid,
            iid=iid,
            num_users=num_users,
            num_items=num_items,
            emb_size=layers[0] / 2,
            reg_user=reg_layers[0],
            reg_item=reg_layers[0],
            seed=seed,
            scope=scope,
        )
        interaction = tf.concat([user_emb, item_emb], axis=-1)
        for i, layer in enumerate(layers[1:]):
            interaction = tf.layers.dense(
                interaction,
                units=layer,
                name="layer{}".format(i + 1),
                activation=act_functions.get(act_fn, tf.nn.relu),
                kernel_initializer=tf.initializers.lecun_uniform(seed),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_layers[i + 1]),
            )
        return interaction

