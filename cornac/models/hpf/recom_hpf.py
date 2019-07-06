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

from cornac.models.hpf import hpf
from ..recommender import Recommender
from ...exception import ScoreException



# HierarchicalPoissonFactorization: Hpf
class HPF(Recommender):
    """Hierarchical Poisson Factorization.

    Parameters
    ----------
    k: int, optional, default: 5
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations.

    name: string, optional, default: 'HPF'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained (Theta and Beta are not None). 
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.
        
    hierarchical: boolean, optional, default: True
        When False, PF is used instead of HPF.

    init_params: dictionary, optional, default: {'G_s':None, 'G_r':None, 'L_s':None, 'L_r':None}
        List of initial parameters, e.g., init_params = {'G_s':G_s, 'G_r':G_r, 'L_s':L_s, 'L_r':L_r}, \
        where G_s and G_r are of type csc_matrix or np.array with the same shape as Theta, see below). \
        They represent respectively the "shape" and "rate" parameters of Gamma distribution over \
        Theta. Similarly, L_s, L_r are the shape and rate parameters of the Gamma over Beta.
      
    Theta: csc_matrix, shape (n_users,k)
        The expected user latent factors.

    Beta: csc_matrix, shape (n_items,k)
        The expected item latent factors.

    References
    ----------
    * Gopalan, Prem, Jake M. Hofman, and David M. Blei. Scalable Recommendation with \
    Hierarchical Poisson Factorization. In UAI, pp. 326-335. 2015.
    """

    def __init__(self, k=5, max_iter=100, name="HPF", trainable=True,
                 verbose=False, hierarchical = True, init_params={'G_s': None, 'G_r': None, 'L_s': None, 'L_r': None}):
        Recommender.__init__(self, name=name, trainable=trainable, verbose = verbose)
        self.k = k
        self.init_params = init_params
        self.max_iter = max_iter

        self.ll = np.full(max_iter, 0)
        self.etp_r = np.full(max_iter, 0)
        self.etp_c = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.hierarchical  = hierarchical
        self.Theta = None  # matrix of user factors
        self.Beta = None  # matrix of item factors


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
            if self.hierarchical:
                res = hpf.hpf(tX, X.shape[0], X.shape[1], self.k, self.max_iter, self.init_params)
            else:
                res = hpf.pf(tX, X.shape[0], X.shape[1], self.k, self.max_iter, self.init_params)
            self.Theta = np.asarray(res['Z'])
            self.Beta = np.asarray(res['W'])
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
                u_representation = np.ones(self.k)
            else:
                u_representation = self.Theta[user_id, :]

            known_item_scores = self.Beta.dot(u_representation)
            known_item_scores = np.array(known_item_scores, dtype='float64').flatten()
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))

            user_pred = self.Beta[item_id,:].dot(self.Theta[user_id, :])
            user_pred = np.array(user_pred, dtype='float64').flatten()[0]

            return user_pred