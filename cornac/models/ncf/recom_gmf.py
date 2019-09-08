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


class GMF(Recommender):
    """Generalized Matrix Factorization.

    Parameters
    ----------
    num_factors: int, optional, default: 8
        Embedding size of MF model.

    regs: float, optional, default: 0.
        Regularization for user and item embeddings.

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

    name: string, optional, default: 'GMF'
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

    def __init__(self, name='GMF',
                 num_factors=8, regs=[0., 0.], num_epochs=20, batch_size=256, num_neg=4,
                 lr=0.001, learner='adam', trainable=True, verbose=True, seed=None):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.num_factors = num_factors
        self.regs = regs
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.learning_rate = lr
        self.learner = learner
        self.seed = seed

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
            self._fit_gmf()

        return self

    def _fit_gmf(self):
        import os
        import tensorflow as tf
        from tqdm import trange
        from .ops import gmf, loss_fn, train_fn

        np.random.seed(self.seed)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        graph = tf.Graph()
        with graph.as_default():
            tf.set_random_seed(self.seed)

            self.user_id = tf.placeholder(shape=[None, ], dtype=tf.int32, name='user_id')
            self.item_id = tf.placeholder(shape=[None, ], dtype=tf.int32, name='item_id')
            self.labels = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='labels')

            self.interaction = gmf(uid=self.user_id, iid=self.item_id, num_users=self.train_set.num_users,
                                   num_items=self.train_set.num_items, emb_size=self.num_factors,
                                   reg_user=self.regs[0], reg_item=self.regs[1], seed=self.seed)

            logits = tf.layers.dense(self.interaction, units=1, name='logits',
                                     kernel_initializer=tf.initializers.lecun_uniform(self.seed))
            self.prediction = tf.nn.sigmoid(logits)

            loss = loss_fn(labels=self.labels, logits=logits)
            train_op = train_fn(loss, learning_rate=self.learning_rate, learner=self.learner)

            initializer = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)
        self.sess.run(initializer)

        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            for i, (batch_users, batch_items, batch_ratings) in enumerate(
                    self.train_set.uir_iter(self.batch_size, shuffle=True, num_zeros=self.num_neg)):
                _, _loss = self.sess.run([train_op, loss],
                                         feed_dict={
                                             self.user_id: batch_users,
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
                self.user_id: [user_idx], self.item_id: np.arange(self.train_set.num_items)
            })
            return known_item_scores.ravel()
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(item_idx):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_idx, item_idx))

            user_pred = self.sess.run(self.prediction, feed_dict={
                self.user_id: [user_idx], self.item_id: [item_idx]
            })
            return user_pred.ravel()
