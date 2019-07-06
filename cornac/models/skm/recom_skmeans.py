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

from ..recommender import Recommender
from ...exception import ScoreException


class SKMeans(Recommender):
    """Spherical k-means based recommender.

    Parameters
    ----------
    k: int, optional, default: 5
        The number of clusters.

    max_iter: int, optional, default: 100
        Maximum number of iterations.

    name: string, optional, default: 'Skmeans'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        trained. 
        
    tol : float, optional, default: 1e-6
        Relative tolerance with regards to skmeans' criterion to declare convergence.
        
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_par: numpy 1d array, optional, default: None
        The initial object parition, 1d array contaning the cluster label (int type starting from 0) \
        of each object (user). If par = None, then skmeans is initialized randomly.
      
    centroids: csc_matrix, shape (k,n_users)
        The maxtrix of cluster centroids.

    References
    ----------
    * Salah, Aghiles, Nicoleta Rogovschi, and Mohamed Nadif. "A dynamic collaborative filtering system \
    via a weighted clustering approach." Neurocomputing 175 (2016): 206-215.
    """

    def __init__(self, k=5, max_iter=100, name="Skmeans", trainable=True, tol=1e-6, verbose=True, init_par=None):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.init_par = init_par
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.centroids = None  # matrix of cluster centroids

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
        from .skmeans import skmeans

        Recommender.fit(self, train_set)

        X = self.train_set.matrix
        X = sp.csr_matrix(X)

        # Skmeans requires rows of X to have a unit L2 norm. We therefore need to make a copy of X as we should not modify the latter.
        X1 = X.copy()
        X1 = X1.multiply(sp.csc_matrix(1. / (np.sqrt(X1.multiply(X1).sum(1).A1) + 1e-20)).T)

        if self.trainable:
            res = skmeans(X1, k=self.k, max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
                          init_par=self.init_par)
            self.centroids = res['centroids']
            self.final_par = res['partition']
        else:
            print('%s is trained already (trainable = False)' % (self.name))
        self.user_center_sim = X1 * self.centroids.T  # user-centroid cosine similarity matrix
        del (X1)

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

            known_item_scores = self.centroids.multiply(self.user_center_sim[user_id, :].T)
            known_item_scores = known_item_scores.sum(0).A1 / (
                    self.user_center_sim[user_id, :].sum() + 1e-20)  # weighted average of cluster centroids
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))

            user_pred = self.centroids[item_id, :].multiply(self.user_center_sim[user_id, :].T)
            # transform user_pred to a flatten array
            user_pred = user_pred.sum(0).A1 / (
                    self.user_center_sim[user_id, :].sum() + 1e-20)  # weighted average of cluster centroids

            return user_pred
