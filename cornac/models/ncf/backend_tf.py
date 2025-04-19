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


def get_optimizer(learning_rate, learner):
    if learner.lower() == "adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif learner.lower() == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif learner.lower() == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)


class GMFLayer(tf.keras.layers.Layer):
    def __init__(self, num_users, num_items, emb_size, reg_user, reg_item, seed=None, **kwargs):
        super(GMFLayer, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.emb_size = emb_size
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.seed = seed
        
        # Initialize embeddings
        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            emb_size,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=seed),
            embeddings_regularizer=tf.keras.regularizers.L2(reg_user),
            name="user_embedding"
        )
        
        self.item_embedding = tf.keras.layers.Embedding(
            num_items,
            emb_size,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=seed),
            embeddings_regularizer=tf.keras.regularizers.L2(reg_item),
            name="item_embedding"
        )
    
    def call(self, inputs):
        user_ids, item_ids = inputs
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return tf.multiply(user_emb, item_emb)


class MLPLayer(tf.keras.layers.Layer):
    def __init__(self, num_users, num_items, layers, reg_layers, act_fn, seed=None, **kwargs):
        super(MLPLayer, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        self.reg_layers = reg_layers
        self.act_fn = act_fn
        self.seed = seed
        
        # Initialize embeddings
        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            int(layers[0] / 2),
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=seed),
            embeddings_regularizer=tf.keras.regularizers.L2(reg_layers[0]),
            name="user_embedding"
        )
        
        self.item_embedding = tf.keras.layers.Embedding(
            num_items,
            int(layers[0] / 2),
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=seed),
            embeddings_regularizer=tf.keras.regularizers.L2(reg_layers[0]),
            name="item_embedding"
        )
        
        # Define dense layers
        self.dense_layers = []
        for i, layer_size in enumerate(layers[1:]):
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    layer_size,
                    activation=act_functions.get(act_fn, tf.nn.relu),
                    kernel_initializer=tf.keras.initializers.LecunUniform(seed=seed),
                    kernel_regularizer=tf.keras.regularizers.L2(reg_layers[i + 1]),
                    name=f"layer{i+1}"
                )
            )
    
    def call(self, inputs):
        user_ids, item_ids = inputs
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        interaction = tf.concat([user_emb, item_emb], axis=-1)
        
        for layer in self.dense_layers:
            interaction = layer(interaction)
            
        return interaction
