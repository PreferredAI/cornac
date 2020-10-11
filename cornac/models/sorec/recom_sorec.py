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
import scipy

from ..recommender import Recommender
from ...utils.common import sigmoid
from ...utils.common import scale
from ...exception import ScoreException


class SoRec(Recommender):
    """Social recommendation using Probabilistic Matrix Factorization.

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

    lamda_c: float, optional, default: 10
        The parameter balancing the information from the user-item rating matrix and the user social network.

    weight_link: boolean, optional, default: True
        When true the social network links are weighted according to eq. (4) in the original paper.

    name: string, optional, default: 'SOREC'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already
        pre-trained (U, V and Z are not None).

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V, 'Z':Z}.

        U: a ndarray of shape (n_users, k)
            Containing the user latent factors.
            
        V: a ndarray of shape (n_items, k)
            Containing the item latent factors.
            
        Z: a ndarray of shape (n_users, k)
            Containing the social network latent factors.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * H. Ma, H. Yang, M. R. Lyu, and I. King. SoRec:Social recommendation using probabilistic matrix factorization. \
     CIKM ’08, pages 931–940, Napa Valley, USA, 2008.

    """

    def __init__(
        self,
        name="SoRec",
        k=5,
        max_iter=100,
        learning_rate=0.001,
        lamda_c=10,
        lamda=0.001,
        gamma=0.9,
        weight_link=True,
        trainable=True,
        verbose=False,
        init_params=None,
        seed=None,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lamda_c = lamda_c
        self.lamda = lamda
        self.gamma = gamma
        self.weight_link = weight_link

        self.ll = np.full(max_iter, 0)
        self.eps = 0.000000001
        self.seed = seed

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)  # matrix of user factors
        self.V = self.init_params.get("V", None)  # matrix of item factors
        self.Z = self.init_params.get("Z", None)  # matrix of social network factors

        if self.U is not None and self.U.shape[1] != self.k:
            raise ValueError("initial parameters U dimension error")

        if self.V is not None and self.V.shape[1] != self.k:
            raise ValueError("initial parameters V dimension error")

        if self.Z is not None and self.Z.shape[1] != self.k:
            raise ValueError("initial parameters Z dimension error")

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

        import math
        from cornac.models.sorec import sorec

        if self.trainable:
            # user-item interactions
            (rat_uid, rat_iid, rat_val) = train_set.uir_tuple

            # user social network
            map_uid = train_set.user_indices
            (net_uid, net_jid, net_val) = train_set.user_graph.get_train_triplet(
                map_uid, map_uid
            )

            if self.weight_link:
                degree = train_set.user_graph.get_node_degree(map_uid, map_uid)
                weighted_net_val = []
                for u, j, val in zip(net_uid, net_jid, net_val):
                    u_out = degree[int(u)][1]
                    j_in = degree[int(j)][0]
                    val_weighted = math.sqrt(j_in / (j_in + u_out)) * val
                    weighted_net_val.append(val_weighted)
                net_val = weighted_net_val

            if [self.train_set.min_rating, self.train_set.max_rating] != [0, 1]:
                if self.train_set.min_rating == self.train_set.max_rating:
                    rat_val = scale(rat_val, 0.0, 1.0, 0.0, self.train_set.max_rating)
                else:
                    rat_val = scale(
                        rat_val,
                        0.0,
                        1.0,
                        self.train_set.min_rating,
                        self.train_set.max_rating,
                    )

            rat_val = np.array(rat_val, dtype="float32")
            rat_uid = np.array(rat_uid, dtype="int32")
            rat_iid = np.array(rat_iid, dtype="int32")

            net_val = np.array(net_val, dtype="float32")
            net_uid = np.array(net_uid, dtype="int32")
            net_jid = np.array(net_jid, dtype="int32")

            if self.verbose:
                print("Learning...")

            res = sorec.sorec(
                rat_uid,
                rat_iid,
                rat_val,
                net_uid,
                net_jid,
                net_val,
                k=self.k,
                n_users=train_set.num_users,
                n_items=train_set.num_items,
                n_ratings=len(rat_val),
                n_edges=len(net_val),
                n_epochs=self.max_iter,
                lamda_c=self.lamda_c,
                lamda=self.lamda,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                init_params={"U": self.U, "V": self.V, "Z": self.Z},
                verbose=self.verbose,
                seed=self.seed,
            )

            self.U = np.asarray(res["U"])
            self.V = np.asarray(res["V"])
            self.Z = np.asarray(res["Z"])

            if self.verbose:
                print("Learning completed")
                
        elif self.verbose:
            print("%s is trained already (trainable = False)" % self.name)

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
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            known_item_scores = self.V.dot(self.U[user_idx, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = self.V[item_idx, :].dot(self.U[user_idx, :])
            user_pred = sigmoid(user_pred)
            if self.train_set.min_rating == self.train_set.max_rating:
                user_pred = scale(user_pred, 0.0, self.train_set.max_rating, 0.0, 1.0)
            else:
                user_pred = scale(
                    user_pred,
                    self.train_set.min_rating,
                    self.train_set.max_rating,
                    0.0,
                    1.0,
                )

            return user_pred
