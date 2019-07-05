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
from ...utils.common import sigmoid
from ...utils.common import scale
from ...exception import ScoreException


class VMF(Recommender):
    """Visual Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the user and item factors.
        
    d: int, optional, default: 10
       The dimension of the user visual factors.

    n_epochs: int, optional, default: 100
        The number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD_RMSProp.

    gamma: float, optional, default: 0.9
        The weight for previous/current gradient in RMSProp.

    lambda_u: float, optional, default: 0.001
        The regularization parameter for user factors.

    lambda_v: float, optional, default: 0.001
        The regularization parameter for item factors.

    lambda_p: float, optional, default: 1.0
        The regularization parameter for user visual factors.

    lambda_e: float, optional, default: 10.
        The regularization parameter for the kernel embedding matrix
        
    lambda_u: float, optional, default: 0.001
        The regularization parameter for user factors.

    name: string, optional, default: 'VMF'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained (The parameters of the model U, V, P, E are not None).
        
    visual_features : See "cornac/examples/vmf_example.py" for an example of how to use \
        cornac's visual module to load and provide the ``item visual features'' for VMF.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: {}
        List of initial parameters, e.g., init_params = {'U':U, 'V':V, 'P': P, 'E': E}. \
        U: numpy array of shape (n_users,k), user latent factors. \
        V: numpy array of shape (n_items,k), item latent factors.
        P: numpy array of shape (n_users,d), user visual latent factors.
        E: numpy array of shape (d,c), embedding kernel matrix.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Park, Chanyoung, Donghyun Kim, Jinoh Oh, and Hwanjo Yu. "Do Also-Viewed Products Help User Rating Prediction?."\
     In Proceedings of WWW, pp. 1113-1122. 2017.
    """

    def __init__(self, name="VMF", k=10, d=10, n_epochs=100, batch_size=100, learning_rate=0.001, gamma=0.9,
                 lambda_u=0.001, lambda_v=0.001, lambda_p=1., lambda_e=10.,
                 trainable=True, verbose=False, use_gpu=False,
                 init_params={}, seed=None):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.d = d
        self.batch_size = batch_size
        self.init_params = init_params
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_p = lambda_p
        self.lambda_e = lambda_e
        self.use_gpu = use_gpu
        self.loss = np.full(n_epochs, 0)
        self.eps = 0.000000001
        self.U = self.init_params.get('U')  # user factors
        self.V = self.init_params.get('V')  # item factors
        self.P = self.init_params.get('P')  # user visual factors
        self.E = self.init_params.get('E')  # Kernel embedding matrix
        self.seed = seed

    # fit the recommender model to the traning data
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

        if self.trainable:

            # Item visual cnn-features
            self.item_features = train_set.item_image.features[:self.train_set.num_items]

            if self.verbose:
                print('Learning...')

            from .vmf import vmf
            res = vmf(self.train_set, self.item_features, k=self.k, d=self.d, n_epochs=self.n_epochs,
                      batch_size=self.batch_size,
                      lambda_u=self.lambda_u, lambda_v=self.lambda_v, lambda_p=self.lambda_p,
                      lambda_e=self.lambda_e, learning_rate=self.learning_rate, gamma=self.gamma,
                      init_params=self.init_params, use_gpu=self.use_gpu, verbose=self.verbose, seed=self.seed)

            self.U = res['U']
            self.V = res['V']
            self.P = res['P']
            self.E = res['E']
            self.Q = res['Q']

            if self.verbose:
                print('Learning completed')
        elif self.verbose:
            print('%s is trained already (trainable = False)' % (self.name))

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

            known_item_scores = self.V.dot(self.U[user_id, :]) + self.Q.dot(self.P[user_id, :])
            # known_item_scores = np.asarray(np.zeros(self.V.shape[0]),dtype='float32')
            # fast_dot(self.U[user_id], self.V, known_item_scores)
            # fast_dot(self.P[user_id], self.Q, known_item_scores)
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
            user_pred = self.V[item_id, :].dot(self.U[user_id, :]) + self.Q[item_id, :].dot(self.P[user_id, :])
            user_pred = sigmoid(user_pred)

            user_pred = scale(user_pred, self.train_set.min_rating, self.train_set.max_rating, 0., 1.)

            return user_pred
