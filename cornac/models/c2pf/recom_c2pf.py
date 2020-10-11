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
        cornac's graph modality to load and provide "item context" for C2PF.

    init_params: dict, optional, default: None
        List of initial parameters, e.g., init_params = {'G_s':G_s, 'G_r':G_r, 'L_s':L_s, 'L_r':L_r, 'L2_s':L2_s, 'L2_r':L2_r, 'L3_s':L3_s, 'L3_r': L3_r}

        Theta: ndarray, shape (n_users, k)
            The expected user latent factors.

        Beta: ndarray, shape (n_items, k)
            The expected item latent factors.

        Xi: ndarray, shape (n_items, k)
            The expected context item latent factors multiplied by context effects Kappa.
            
        G_s: ndarray, shape (n_users, k)
            Represent the "shape" parameters of Gamma distribution over Theta.
            
        G_r: ndarray, shape (n_users, k)
            Represent the "rate" parameters of Gamma distribution over Theta. 
        
        L_s: ndarray, shape (n_items, k)
            Represent the "shape" parameters of Gamma distribution over Beta.
            
        L_r: ndarray, shape (n_items, k)
            Represent the "rate" parameters of Gamma distribution over Beta. 
        
        L2_s: ndarray, shape (n_items, k)
            Represent the "shape" parameters of Gamma distribution over Xi.
            
        L2_r: ndarray, shape (n_items, k)
            Represent the "rate" parameters of Gamma distribution over Xi.
             
        L3_s: ndarray
            Represent the "shape" parameters of Gamma distribution over Kappa.
            
        L3_r: ndarray 
            Represent the "rate" parameters of Gamma distribution over Kappa.

    References
    ----------
    * Salah, Aghiles, and Hady W. Lauw. A Bayesian Latent Variable Model of User Preferences with Item Context. \
    In IJCAI, pp. 2667-2674. 2018.
    """

    def __init__(
        self,
        k=100,
        max_iter=100,
        variant="c2pf",
        name=None,
        trainable=True,
        verbose=False,
        init_params=None,
    ):
        if name is None:
            Recommender.__init__(
                self, name=variant.upper(), trainable=trainable, verbose=verbose
            )
        else:
            Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)

        self.k = k
        self.max_iter = max_iter

        self.ll = np.full(max_iter, 0)
        self.eps = 0.000000001
        # self.aux_info = aux_info  # item-context matrix in the triplet sparse format: (row_id, col_id, value)
        self.variant = variant

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.Theta = self.init_params.get("Theta", None)
        self.Beta = self.init_params.get("Beta", None)
        self.Xi = self.init_params.get("Xi", None)
        self.Gs = self.init_params.get("G_s", None)
        self.Gr = self.init_params.get("G_r", None)
        self.Ls = self.init_params.get("L_s", None)
        self.Lr = self.init_params.get("L_r", None)
        self.L2s = self.init_params.get("L2_s", None)
        self.L2r = self.init_params.get("L2_r", None)
        self.L3s = self.init_params.get("L3_s", None)
        self.L3r = self.init_params.get("L3_r", None)

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

        if self.trainable:
            # use pre-trained params if exists, otherwise from constructor
            init_params = {
                "G_s": self.Gs,
                "G_r": self.Gr,
                "L_s": self.Ls,
                "L_r": self.Lr,
                "L2_s": self.L2s,
                "L2_r": self.L2r,
                "L3_s": self.L3s,
                "L3_r": self.L3r,
            }

            map_iid = train_set.item_indices
            (rid, cid, val) = train_set.item_graph.get_train_triplet(map_iid, map_iid)
            context_info = np.hstack(
                (rid.reshape(-1, 1), cid.reshape(-1, 1), val.reshape(-1, 1))
            )

            if self.variant == "c2pf":
                res = c2pf.c2pf(
                    tX,
                    X.shape[0],
                    X.shape[1],
                    context_info,
                    X.shape[1],
                    X.shape[1],
                    self.k,
                    self.max_iter,
                    init_params,
                )
            elif self.variant == "tc2pf":
                res = c2pf.t_c2pf(
                    tX,
                    X.shape[0],
                    X.shape[1],
                    context_info,
                    X.shape[1],
                    X.shape[1],
                    self.k,
                    self.max_iter,
                    init_params,
                )
            elif self.variant == "rc2pf":
                res = c2pf.r_c2pf(
                    tX,
                    X.shape[0],
                    X.shape[1],
                    context_info,
                    X.shape[1],
                    X.shape[1],
                    self.k,
                    self.max_iter,
                    init_params,
                )
            else:
                res = c2pf.c2pf(
                    tX,
                    X.shape[0],
                    X.shape[1],
                    context_info,
                    X.shape[1],
                    X.shape[1],
                    self.k,
                    self.max_iter,
                    init_params,
                )

            self.Theta = sp.csc_matrix(res["Z"]).todense()
            self.Beta = sp.csc_matrix(res["W"]).todense()
            self.Xi = sp.csc_matrix(res["Q"]).todense()

            # overwrite init_params for future fine-tuning
            self.Gs = np.asarray(res["G_s"])
            self.Gr = np.asarray(res["G_r"])
            self.Ls = np.asarray(res["L_s"])
            self.Lr = np.asarray(res["L_r"])
            self.L2s = np.asarray(res["L2_s"])
            self.L2r = np.asarray(res["L2_r"])
            self.L3s = np.asarray(res["L3_s"])
            self.L3r = np.asarray(res["L3_r"])

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
        if self.variant == "c2pf" or self.variant == "tc2pf":
            if item_idx is None:
                user_pred = (
                    self.Beta * self.Theta[user_idx, :].T
                    + self.Xi * self.Theta[user_idx, :].T
                )
            else:
                user_pred = (
                    self.Beta[item_idx, :] * self.Theta[user_idx, :].T
                    + self.Xi * self.Theta[user_idx, :].T
                )
        elif self.variant == "rc2pf":
            if item_idx is None:
                user_pred = self.Xi * self.Theta[user_idx, :].T
            else:
                user_pred = self.Xi[item_idx,] * self.Theta[user_idx, :].T
        else:
            if item_idx is None:
                user_pred = (
                    self.Beta * self.Theta[user_idx, :].T
                    + self.Xi * self.Theta[user_idx, :].T
                )
            else:
                user_pred = (
                    self.Beta[item_idx, :] * self.Theta[user_idx, :].T
                    + self.Xi * self.Theta[user_idx, :].T
                )
        # transform user_pred to a flatten array,
        user_pred = np.array(user_pred, dtype="float64").flatten()

        return user_pred
