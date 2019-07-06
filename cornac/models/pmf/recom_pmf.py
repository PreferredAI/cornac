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


class PMF(Recommender):
    """Probabilistic Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 5
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD_RMSProp.
        
    gamma: float, optional, default: 0.9
        The weight for previous/current gradient in RMSProp.

    lamda: float, optional, default: 0.001
        The regularization parameter.

    name: string, optional, default: 'PMF'
        The name of the recommender model.
        
    variant: {"linear","non_linear"}, optional, default: 'non_linear'
        Pmf variant. If 'non_linear', the Gaussian mean is the output of a Sigmoid function.\
        If 'linear' the Gaussian mean is the output of the identity function.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: {}
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}. \
        U: a csc_matrix of shape (n_users,k), containing the user latent factors. \
        V: a csc_matrix of shape (n_items,k), containing the item latent factors.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Mnih, Andriy, and Ruslan R. Salakhutdinov. Probabilistic matrix factorization. \
    In NIPS, pp. 1257-1264. 2008.
    """

    def __init__(self, k=5, max_iter=100, learning_rate=0.001, gamma=0.9, lamda=0.001, name="PMF", variant='non_linear',
                 trainable=True, verbose=False, init_params={}, seed=None):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lamda = lamda
        self.variant = variant

        self.ll = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.U = self.init_params.get('U')  # matrix of user factors
        self.V = self.init_params.get('V')  # matrix of item factors
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
        from cornac.models.pmf import pmf

        Recommender.fit(self, train_set)

        if self.trainable:
            # converting data to the triplet format (needed for cython function pmf)
            (uid, iid, rat) = train_set.uir_tuple
            rat = np.array(rat, dtype='float32')
            if self.variant == 'non_linear':  # need to map the ratings to [0,1]
                if [self.train_set.min_rating, self.train_set.max_rating] != [0, 1]:
                    rat = scale(rat, 0., 1., self.train_set.min_rating, self.train_set.max_rating)
            uid = np.array(uid, dtype='int32')
            iid = np.array(iid, dtype='int32')

            if self.verbose:
                print('Learning...')

            if self.variant == 'linear':
                res = pmf.pmf_linear(uid, iid, rat, k=self.k, n_users=train_set.num_users, n_items=train_set.num_items,
                                     n_ratings=len(rat), n_epochs=self.max_iter,
                                     lamda=self.lamda, learning_rate=self.learning_rate, gamma=self.gamma,
                                     init_params=self.init_params, verbose=self.verbose, seed=self.seed)
            elif self.variant == 'non_linear':
                res = pmf.pmf_non_linear(uid, iid, rat, k=self.k, n_users=train_set.num_users,
                                         n_items=train_set.num_items, n_ratings=len(rat), n_epochs=self.max_iter,
                                         lamda=self.lamda, learning_rate=self.learning_rate, gamma=self.gamma,
                                         init_params=self.init_params, verbose=self.verbose, seed=self.seed)
            else:
                raise ValueError('variant must be one of {"linear","non_linear"}')

            self.U = np.asarray(res['U'])
            self.V = np.asarray(res['V'])

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

            known_item_scores = self.V.dot(self.U[user_id, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))

            user_pred = self.V[item_id, :].dot(self.U[user_id, :])

            if self.variant == "non_linear":
                user_pred = sigmoid(user_pred)
                user_pred = scale(user_pred, self.train_set.min_rating, self.train_set.max_rating, 0., 1.)

            return user_pred
