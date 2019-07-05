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


class CF(Recommender):
    """Collaborative Filtering for Implicit Feedbacks

    Parameters
    ----------
    name: string, default: 'CF'
        The name of the recommender model.

    k: int, optional, default: 200
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for AdamOptimizer.

    lambda_u: float, optional, default: 0.01
        The regularization parameter for users.

    lambda_v: float, optional, default: 0.01
        The regularization parameter for items.

    a: float, optional, default: 1
        The confidence of observed ratings.

    b: float, optional, default: 0.01
        The confidence of unseen ratings.

    batch_size: int, optional, default: 128
        The batch size for SGD.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already 
        pre-trained (U and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}
        U: ndarray, shape (n_users,k)
            The user latent factors, optional initialization via init_params.
        V: ndarray, shape (n_items,k)
            The item latent factors, optional initialization via init_params.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    * Yifan Hu, Yehuda Koren∗, Chris Volinsky. CF: Collaborative Filtering for Implicit Feedbacks.
    In : 2008 Eighth IEEE International Conference on Data Mining
    """

    def __init__(self, name='CF', k=200, lambda_u=0.01, lambda_v=0.01,
                 a=1, b=0.01, learning_rate=0.001, batch_size=128, max_iter=100,
                 trainable=True, verbose=True, init_params=None, seed=None):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.a = a
        self.b = b
        self.learning_rate = learning_rate
        self.name = name
        self.init_params = init_params
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.init_params = {} if not init_params else init_params
        self.seed = seed

    def fit(self, train_set):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: object of type TrainSet, required
            An object contraining the user-item preference in csr scipy sparse format,\
            as well as some useful attributes such as mappings to the original user/item ids.\
            Please refer to the class TrainSet in the "data" module for details.
        """
        Recommender.fit(self, train_set)

        from ...utils import get_rng
        from ...utils.init_utils import xavier_uniform

        self.seed = get_rng(self.seed)
        self.U = self.init_params.get('U', xavier_uniform((self.train_set.num_users, self.k), self.seed))
        self.V = self.init_params.get('V', xavier_uniform((self.train_set.num_items, self.k), self.seed))

        if self.trainable:
            self._fit_cf()

    def _fit_cf(self, ):
        import tensorflow as tf
        from tqdm import trange
        from .cf import Model

        R = self.train_set.csc_matrix  # csc for efficient slicing over items
        n_users, n_items, = self.train_set.num_users, self.train_set.num_items

        # Build model
        model = Model(n_users=n_users, n_items=n_items, k=self.k,
                      lambda_u=self.lambda_u, lambda_v=self.lambda_v,
                      lr=self.learning_rate, U=self.U, V=self.V)

        # Training model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            loop = trange(self.max_iter, disable=not self.verbose)
            for _ in loop:

                sum_loss = 0
                count = 0
                for i, batch_ids in enumerate(self.train_set.item_iter(self.batch_size, shuffle=True)):
                    batch_R = R[:, batch_ids]
                    batch_C = np.ones(batch_R.shape) * self.b
                    batch_C[batch_R.nonzero()] = self.a
                    feed_dict = {
                        model.ratings: batch_R.A,
                        model.C: batch_C,
                        model.item_ids: batch_ids
                    }
                    _, _loss = sess.run([model.opt, model.loss], feed_dict)  # train U, V

                    sum_loss += _loss
                    count += len(batch_ids)
                    if i % 10 == 0:
                        loop.set_postfix(loss=(sum_loss / count))

            self.U, self.V = sess.run([model.U, model.V])

        tf.reset_default_graph()

        if self.verbose:
            print('Learning completed!')

    def score(self, user_id, item_id=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score prediction.

        item_id: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if item_id is None:
            if self.train_set.is_unk_user(user_id):
                raise ScoreException("Can't make score prediction for (user_id=%d)" % user_id)

            known_item_scores = self.V.dot(self.U[user_id, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
            user_pred = self.V[item_id, :].dot(self.U[user_id, :])
            return user_pred
