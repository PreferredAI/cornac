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

from ..recommender import Recommender
from ...exception import ScoreException


class NeuMF(Recommender):
    """Neural Matrix Factorization.

    Parameters
    ----------
    num_factors: int, optional, default: 8
        Embedding size of MF model.

    layers: list, optional, default: [64, 32, 16, 8]
        MLP layers. Note that the first layer is the concatenation of
        user and item embeddings. So layers[0]/2 is the embedding size.

    act_fn: str, default: 'relu'
        Name of the activation function used for the MLP layers.
        Supported functions: ['sigmoid', 'tanh', 'elu', 'relu', 'selu, 'relu6', 'leaky_relu']

    reg_mf: float, optional, default: 0.
        Regularization for MF embeddings.

    reg_layers: list, optional, default: [0., 0., 0., 0.]
        Regularization for each MLP layer,
        reg_layers[0] is the regularization for embeddings.

    num_epochs: int, optional, default: 20
        Number of epochs.

    batch_size: int, optional, default: 256
        Batch size.

    num_neg: int, optional, default: 4
        Number of negative instances to pair with a positive instance.

    lr: float, optional, default: 0.001
        Learning rate.

    learner: str, optional, default: 'adam'
        Specify an optimizer: adagrad, adam, rmsprop, sgd

    name: string, optional, default: 'NeuMF'
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. \
    In Proceedings of the 26th international conference on world wide web (pp. 173-182).
    """

    def __init__(self, name='NeuMF',
                 num_factors=8, layers=[64, 32, 16, 8], act_fn='relu',
                 reg_mf=0., reg_layers=[0., 0., 0., 0.], num_epochs=20,
                 batch_size=256, num_neg=4, lr=0.001, learner='adam',
                 trainable=True, verbose=True, seed=None):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.num_factors = num_factors
        self.layers = layers
        self.act_fn = act_fn
        self.reg_mf = reg_mf
        self.reg_layers = reg_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.learning_rate = lr
        self.learner = learner
        self.seed = seed
        self.pretrained = False

    def pretrain(self, gmf_model, mlp_model, alpha=0.5):
        """Provide pre-trained GMF and MLP models. Section 3.4.1 of the paper.

        Parameters
        ----------
        gmf_model: object of type GMF, required
            Reference to trained/fitted GMF model.

        gmf_model: object of type GMF, required
            Reference to trained/fitted GMF model.

        alpha: float, optional, default: 0.5
            Hyper-parameter determining the trade-off between the two pre-trained models.
            Details are described in the section 3.4.1 of the paper.
        """
        self.pretrained = True
        self.gmf_model = gmf_model
        self.mlp_model = mlp_model
        self.alpha = alpha
        return self

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            self._fit_neumf()

        return self

    def _fit_neumf(self):
        import os
        import tensorflow as tf
        from tqdm import trange
        from .ops import gmf, mlp, loss_fn, train_fn

        np.random.seed(self.seed)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(self.seed)

            self.gmf_user_id = tf.placeholder(shape=[None, ], dtype=tf.int32, name='gmf_user_id')
            self.mlp_user_id = tf.placeholder(shape=[None, ], dtype=tf.int32, name='mlp_user_id')
            self.item_id = tf.placeholder(shape=[None, ], dtype=tf.int32, name='item_id')
            self.labels = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='labels')

            gmf_feat = gmf(uid=self.gmf_user_id, iid=self.item_id, num_users=self.train_set.num_users,
                           num_items=self.train_set.num_items, emb_size=self.num_factors,
                           reg_user=self.reg_mf, reg_item=self.reg_mf, seed=self.seed)
            mlp_feat = mlp(uid=self.mlp_user_id, iid=self.item_id, num_users=self.train_set.num_users,
                           num_items=self.train_set.num_items, layers=self.layers,
                           reg_layers=self.reg_layers, act_fn=self.act_fn, seed=self.seed)

            interaction = tf.concat([gmf_feat, mlp_feat], axis=-1)
            logits = tf.layers.dense(interaction, units=1, name='logits',
                                     kernel_initializer=tf.initializers.lecun_uniform(self.seed))
            self.prediction = tf.nn.sigmoid(logits)

            loss = loss_fn(labels=self.labels, logits=logits)
            train_op = train_fn(loss, learning_rate=self.learning_rate, learner=self.learner)

            initializer = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        self.sess.run(initializer)

        if self.pretrained:
            gmf_kernel = self.gmf_model.sess.run(self.gmf_model.sess.graph.get_tensor_by_name('logits/kernel:0'))
            gmf_bias = self.gmf_model.sess.run(self.gmf_model.sess.graph.get_tensor_by_name('logits/bias:0'))
            mlp_kernel = self.mlp_model.sess.run(self.mlp_model.sess.graph.get_tensor_by_name('logits/kernel:0'))
            mlp_bias = self.mlp_model.sess.run(self.mlp_model.sess.graph.get_tensor_by_name('logits/bias:0'))
            logits_kernel = np.concatenate([self.alpha * gmf_kernel, (1 - self.alpha) * mlp_kernel])
            logits_bias = self.alpha * gmf_bias + (1 - self.alpha) * mlp_bias

            for v in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                if v.name.startswith('GMF'):
                    sess = self.gmf_model.sess
                    self.sess.run(tf.assign(v, sess.run(sess.graph.get_tensor_by_name(v.name))))
                elif v.name.startswith('MLP'):
                    sess = self.mlp_model.sess
                    self.sess.run(tf.assign(v, sess.run(sess.graph.get_tensor_by_name(v.name))))
                elif v.name.startswith('logits/kernel'):
                    self.sess.run(tf.assign(v, logits_kernel))
                elif v.name.startswith('logits/bias'):
                    self.sess.run(tf.assign(v, logits_bias))

        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            for i, (batch_users, batch_items, batch_ratings) in enumerate(
                    self.train_set.uir_iter(self.batch_size, shuffle=True, num_zeros=self.num_neg)):
                _, _loss = self.sess.run([train_op, loss], feed_dict={
                    self.gmf_user_id: batch_users,
                    self.mlp_user_id: batch_users,
                    self.item_id: batch_items,
                    self.labels: batch_ratings.reshape(-1, 1)
                })
                count += len(batch_ratings)
                sum_loss += _loss * len(batch_ratings)
                if i % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException("Can't make score prediction for (user_id=%d)" % user_idx)

            known_item_scores = self.sess.run(self.prediction, feed_dict={
                self.gmf_user_id: [user_idx],
                self.mlp_user_id: np.ones(self.train_set.num_items) * user_idx,
                self.item_id: np.arange(self.train_set.num_items)
            })
            return known_item_scores.ravel()
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(item_idx):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_idx, item_idx))

            user_pred = self.sess.run(self.prediction, feed_dict={
                self.gmf_user_id: [user_idx], self.mlp_user_id: [user_idx], self.item_id: [item_idx]
            })
            return user_pred.ravel()
