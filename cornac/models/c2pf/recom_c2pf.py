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
import scipy.sparse as sp

from cornac.models.c2pf import c2pf
from ..recommender import Recommender


# Recommender class for Collaborative Context Poisson Factorization (C2PF)
class C2PF(Recommender):
    """Collaborative Context Poisson Factorization.

    Parameters
    ----------
    k: int, optional, default: 100
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations for variational C2PF.

    variant: string, optional, default: 'c2pf'
        C2pf's variant: c2pf: 'c2pf', 'tc2pf' (tied-c2pf) or 'rc2pf' (reduced-c2pf). \
        Please refer to the original paper for details.

    name: string, optional, default: None
        The name of the recommender model. If None, \
        then "variant" is used as the default name of the model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (Theta, Beta and Xi are not None).

    Item_context: See "cornac/examples/c2pf_example.py" in the GitHub repo for an example of how to use \
        cornac's graph module to load and provide "item context" for C2PF.

    init_params: dictionary, optional, default: {'G_s':None, 'G_r':None, 'L_s':None, 'L_r':None, \
        'L2_s':None, 'L2_r':None, 'L3_s':None, 'L3_r':None}
        List of initial parameters, e.g., init_params = {'G_s':G_s, 'G_r':G_r, 'L_s':L_s, 'L_r':L_r, \
        'L2_s':L2_s, 'L2_r':L2_r, 'L3_s':L3_s, 'L3_r':L3_r}, \
        where G_s and G_r are of type csc_matrix or np.array with the same shape as Theta, see below). \
        They represent respectively the "shape" and "rate" parameters of Gamma distribution over \
        Theta. It is the same for L_s, L_r and Beta, L2_s, L2_r and Xi, L3_s, L3_r and Kappa.

    Theta: csc_matrix, shape (n_users,k)
        The expected user latent factors.

    Beta: csc_matrix, shape (n_items,k)
        The expected item latent factors.

    Xi: csc_matrix, shape (n_items,k)
        The expected context item latent factors multiplied by context effects Kappa, \
        please refer to the paper below for details.

    References
    ----------
    * Salah, Aghiles, and Hady W. Lauw. A Bayesian Latent Variable Model of User Preferences with Item Context. \
    In IJCAI, pp. 2667-2674. 2018.
    """

    def __init__(self, k=100, max_iter=100, variant='c2pf', name=None, trainable=True, verbose=False,
                 init_params={'G_s': None, 'G_r': None, 'L_s': None, 'L_r': None, 'L2_s': None, 'L2_r': None,
                              'L3_s': None, 'L3_r': None}):
        if name is None:
            Recommender.__init__(self, name=variant.upper(), trainable=trainable, verbose=verbose)
        else:
            Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)

        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter

        self.ll = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.Theta = None  # user factors
        self.Beta = None  # item factors
        self.Xi = None  # context factors Xi multiplied by context effects Kappa
        # self.aux_info = aux_info  # item-context matrix in the triplet sparse format: (row_id, col_id, value)
        self.variant = variant

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
        X = sp.csc_matrix(self.train_set.matrix)

        # recover the striplet sparse format from csc sparse matrix X (needed to feed c++)
        (rid, cid, val) = sp.find(X)
        val = np.array(val, dtype='float32')
        rid = np.array(rid, dtype='int32')
        cid = np.array(cid, dtype='int32')
        tX = np.concatenate((np.concatenate(([rid], [cid]), axis=0).T, val.reshape((len(val), 1))), axis=1)
        del rid, cid, val

        if self.trainable:
            map_iid = train_set.iid_list
            (rid, cid, val) = train_set.item_graph.get_train_triplet(map_iid, map_iid)
            context_info = np.hstack((rid.reshape(-1, 1), cid.reshape(-1, 1), val.reshape(-1, 1)))

            if self.variant == 'c2pf':
                res = c2pf.c2pf(tX, X.shape[0], X.shape[1], context_info, X.shape[1], X.shape[1], self.k,
                                self.max_iter,
                                self.init_params)
            elif self.variant == 'tc2pf':
                res = c2pf.t_c2pf(tX, X.shape[0], X.shape[1], context_info, X.shape[1], X.shape[1], self.k,
                                  self.max_iter,
                                  self.init_params)
            elif self.variant == 'rc2pf':
                res = c2pf.r_c2pf(tX, X.shape[0], X.shape[1], context_info, X.shape[1], X.shape[1], self.k,
                                  self.max_iter,
                                  self.init_params)
            else:
                res = c2pf.c2pf(tX, X.shape[0], X.shape[1], context_info, X.shape[1], X.shape[1], self.k,
                                self.max_iter,
                                self.init_params)

            self.Theta = sp.csc_matrix(res['Z']).todense()
            self.Beta = sp.csc_matrix(res['W']).todense()
            self.Xi = sp.csc_matrix(res['Q']).todense()
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
        if self.variant == 'c2pf' or self.variant == 'tc2pf':
            if item_id is None:
                user_pred = self.Beta * self.Theta[user_id, :].T + self.Xi * self.Theta[user_id, :].T
            else:
                user_pred = self.Beta[item_id, :] * self.Theta[user_id, :].T + self.Xi * self.Theta[user_id, :].T
        elif self.variant == 'rc2pf':
            if item_id is None:
                user_pred = self.Xi * self.Theta[user_id, :].T
            else:
                user_pred = self.Xi[item_id,] * self.Theta[user_id, :].T
        else:
            if item_id is None:
                user_pred = self.Beta * self.Theta[user_id, :].T + self.Xi * self.Theta[user_id, :].T
            else:
                user_pred = self.Beta[item_id, :] * self.Theta[user_id, :].T + self.Xi * self.Theta[user_id, :].T
        # transform user_pred to a flatten array,
        user_pred = np.array(user_pred, dtype='float64').flatten()

        return user_pred
