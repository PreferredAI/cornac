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
        
    seed: int, optional, default: None
        Random seed for parameters initialization.

    init_params: dict, optional, default: None
        Initial parameters of the model.
        
        Theta: ndarray, shape (n_users, k)
            The expected user latent factors.
        
        Beta: ndarray, shape (n_items, k)
            The expected item latent factors.

        G_s: ndarray, shape (n_users, k)
            This represents "shape" parameters of Gamma distribution over Theta.

        G_r: ndarray, shape (n_users, k)
            This represents "rate" parameters of Gamma distribution over Theta.
            
        L_s: ndarray, shape (n_items, k)
            This represents "shape" parameters of Gamma distribution over Beta.
        
        L_r: ndarray, shape (n_items, k)
            This represents "rate" parameters of Gamma distribution over Beta.
        
    References
    ----------
    * Gopalan, Prem, Jake M. Hofman, and David M. Blei. Scalable Recommendation with \
    Hierarchical Poisson Factorization. In UAI, pp. 326-335. 2015.
    """

    def __init__(
        self,
        k=5,
        max_iter=100,
        name="HPF",
        trainable=True,
        verbose=False,
        hierarchical=True,
        seed=None,
        init_params=None,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter

        self.ll = np.full(max_iter, 0)
        self.etp_r = np.full(max_iter, 0)
        self.etp_c = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.hierarchical = hierarchical
        self.seed = seed

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.Theta = self.init_params.get("Theta", None)  # matrix of user factors
        self.Beta = self.init_params.get("Beta", None)  # matrix of item factors
        self.Gs = self.init_params.get("G_s", None)
        self.Gr = self.init_params.get("G_r", None)
        self.Ls = self.init_params.get("L_s", None)
        self.Lr = self.init_params.get("L_r", None)

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
            # use pre-trained params if exists, otherwise from constructor
            init_params = {
                "G_s": self.Gs,
                "G_r": self.Gr,
                "L_s": self.Ls,
                "L_r": self.Lr,
            }

            X = sp.csc_matrix(self.train_set.matrix)
            # recover the striplet sparse format from csc sparse matrix X (needed to feed c++)
            (rid, cid, val) = sp.find(X)
            val = np.array(val, dtype="float32")
            rid = np.array(rid, dtype="int32")
            cid = np.array(cid, dtype="int32")
            tX = np.concatenate(
                (np.concatenate(([rid], [cid]), axis=0).T, val.reshape((len(val), 1))),
                axis=1,
            )
            del rid, cid, val

            if self.hierarchical:
                res = hpf.hpf(
                    tX, X.shape[0], X.shape[1], self.k, self.max_iter, self.seed, init_params
                )
            else:
                res = hpf.pf(
                    tX, X.shape[0], X.shape[1], self.k, self.max_iter, self.seed, init_params
                )
            self.Theta = np.asarray(res["Z"])
            self.Beta = np.asarray(res["W"])

            # overwrite init_params for future fine-tuning
            self.Gs = np.asarray(res["G_s"])
            self.Gr = np.asarray(res["G_r"])
            self.Ls = np.asarray(res["L_s"])
            self.Lr = np.asarray(res["L_r"])

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                u_representation = np.ones(self.k)
            else:
                u_representation = self.Theta[user_idx, :]

            known_item_scores = self.Beta.dot(u_representation)
            known_item_scores = np.array(known_item_scores, dtype="float64").flatten()
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = self.Beta[item_idx, :].dot(self.Theta[user_idx, :])
            user_pred = np.array(user_pred, dtype="float64").flatten()[0]

            return user_pred
