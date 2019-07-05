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


class MCF(Recommender):
    """Matrix Co-Factorization.

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

    name: string, optional, default: 'MCF'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained (U and V are not None).

    item-affinity network: See "cornac/examples/mcf_office.py" for an example of how to use \
        cornac's graph module to load and provide the ``item-affinity network'' for MCF.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: {}
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}. \
        U: a csc_matrix of shape (n_users,k), containing the user latent factors. \
        V: a csc_matrix of shape (n_items,k), containing the item latent factors. \
        Z: a csc_matrix of shape (n_items,k), containing the ``Also-Viewed'' item latent factors.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Park, Chanyoung, Donghyun Kim, Jinoh Oh, and Hwanjo Yu. "Do Also-Viewed Products Help User Rating Prediction?."\
     In Proceedings of WWW, pp. 1113-1122. 2017.
    """

    def __init__(self, k=5, max_iter=100, learning_rate=0.001, gamma=0.9, lamda=0.001, name="MCF",
                 trainable=True, verbose=False, init_params={}, seed=None):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lamda = lamda

        self.ll = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.U = self.init_params.get('U')  # matrix of user factors
        self.V = self.init_params.get('V')  # matrix of item factors
        self.Z = self.init_params.get('Z')  # matrix of Also-Viewed item factors
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
        from cornac.models.mcf import mcf
        Recommender.fit(self, train_set)

        if self.trainable:
            # user-item interactions
            (rat_uid, rat_iid, rat_val) = train_set.uir_tuple

            # item-item affinity network
            map_iid = train_set.iid_list
            (net_iid, net_jid, net_val) = train_set.item_graph.get_train_triplet(map_iid, map_iid)
            if [self.train_set.min_rating, self.train_set.max_rating] != [0, 1]:
                if self.train_set.min_rating == self.train_set.max_rating:
                    rat_val = scale(rat_val, 0., 1., 0., self.train_set.max_rating)
                else:
                    rat_val = scale(rat_val, 0., 1., self.train_set.min_rating, self.train_set.max_rating)

            if [min(net_val), max(net_val)] != [0, 1]:
                if min(net_val) == max(net_val):
                    net_val = scale(net_val, 0., 1., 0., max(net_val))
                else:
                    net_val = scale(net_val, 0., 1., min(net_val), max(net_val))

            rat_val = np.array(rat_val, dtype='float32')
            rat_uid = np.array(rat_uid, dtype='int32')
            rat_iid = np.array(rat_iid, dtype='int32')

            net_val = np.array(net_val, dtype='float32')
            net_iid = np.array(net_iid, dtype='int32')
            net_jid = np.array(net_jid, dtype='int32')

            if self.verbose:
                print('Learning...')

            res = mcf.mcf(rat_uid, rat_iid, rat_val, net_iid, net_jid, net_val, k=self.k, n_users=train_set.num_users,
                          n_items=train_set.num_items, n_ratings=len(rat_val), n_edges=len(net_val),
                          n_epochs=self.max_iter,
                          lamda=self.lamda, learning_rate=self.learning_rate, gamma=self.gamma,
                          init_params=self.init_params, verbose=self.verbose, seed=self.seed)

            self.U = np.asarray(res['U'])
            self.V = np.asarray(res['V'])
            self.Z = np.asarray(res['Z'])

            if self.verbose:
                print('Learning completed')
        elif self.verbose:
            print('%s is trained already (trainable = False)' % self.name)

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

            user_pred = sigmoid(user_pred)
            if self.train_set.min_rating == self.train_set.max_rating:
                user_pred = scale(user_pred, 0., self.train_set.max_rating, 0., 1.)
            else:
                user_pred = scale(user_pred, self.train_set.min_rating, self.train_set.max_rating, 0., 1.)

            return user_pred
