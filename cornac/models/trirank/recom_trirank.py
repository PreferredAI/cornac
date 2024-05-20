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
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from ..recommender import Recommender
from ...utils import get_rng
from ...utils.init_utils import uniform
from ...exception import ScoreException


EPS = 1e-10


class TriRank(Recommender):
    """TriRank: Review-aware Explainable Recommendation by Modeling Aspects.

    Parameters
    ----------
    name: string, optional, default: 'TriRank'
        The name of the recommender model.

    alpha: float, optional, default: 1
        The weight of smoothness on user-item relation

    beta: float, optional, default: 1
        The weight of smoothness on item-aspect relation

    gamma: float, optional, default: 1
        The weight of smoothness on user-aspect relation

    eta_U: float, optional, default: 1
        The weight of fitting constraint on users

    eta_P: float, optional, default: 1
        The weight of fitting constraint on items

    eta_A: float, optional, default: 1
        The weight of fitting constraint on aspects

    max_iter: int, optional, default: 100
        Maximum number of iterations to stop online training. If set to `max_iter=-1`, \
        the online training will stop when model parameters are converged.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (R, X, Y, p, a, u are not None).

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'R':R, 'X':X, 'Y':Y, 'p':p, 'a':a, 'u':u}

        R: csr_matrix, shape (n_users, n_items)
            The symmetric normalized of edge weight matrix of user-item relation, optional initialization via init_params

        X: csr_matrix, shape (n_items, n_aspects)
            The symmetric normalized of edge weight matrix of item-aspect relation, optional initialization via init_params

        Y: csr_matrix, shape (n_users, n_aspects)
            The symmetric normalized of edge weight matrix of user-aspect relation, optional initialization via init_params

        p: ndarray, shape (n_items,)
            Initialized item weights, optional initialization via init_params

        a: ndarray, shape (n_aspects,)
            Initialized aspect weights, optional initialization via init_params

        u: ndarray, shape (n_aspects,)
            Initialized user weights, optional initialization via init_params

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    He, Xiangnan, Tao Chen, Min-Yen Kan, and Xiao Chen. 2014. \
    TriRank: Review-aware Explainable Recommendation by Modeling Aspects. \
    In the 24th ACM international on conference on information and knowledge management (CIKM'15). \
    ACM, New York, NY, USA, 1661-1670. DOI: https://doi.org/10.1145/2806416.2806504
    """

    def __init__(
        self,
        name="TriRank",
        alpha=1,
        beta=1,
        gamma=1,
        eta_U=1,
        eta_P=1,
        eta_A=1,
        max_iter=100,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta_U = eta_U
        self.eta_P = eta_P
        self.eta_A = eta_A
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        self.rng = get_rng(seed)

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.R = self.init_params.get("R", None)
        self.X = self.init_params.get("X", None)
        self.Y = self.init_params.get("Y", None)
        self.p = self.init_params.get("p", None)
        self.a = self.init_params.get("a", None)
        self.u = self.init_params.get("u", None)

    def _init(self, train_set):
        # Initialize user, item and aspect rank.
        if self.p is None:
            self.p = uniform(train_set.num_items, random_state=self.rng)
        if self.a is None:
            self.a = uniform(train_set.sentiment.num_aspects, random_state=self.rng)
        if self.u is None:
            self.u = uniform(train_set.num_users, random_state=self.rng)

    def _symmetrical_normalization(self, matrix: csr_matrix):
        row = []
        col = []
        data = []
        row_norm = np.sqrt(matrix.sum(axis=1).A1)
        col_norm = np.sqrt(matrix.sum(axis=0).A1)
        for i, j in zip(*matrix.nonzero()):
            row.append(i)
            col.append(j)
            data.append(matrix[i, j] / (row_norm[i] * col_norm[j]))

        return csr_matrix((data, (row, col)), shape=matrix.shape)

    def _create_matrices(self, train_set):
        from time import time

        self.r_mat = train_set.csr_matrix

        start_time = time()
        if self.verbose:
            print("Building matrices started!")
        sentiment_modality = train_set.sentiment
        n_users = train_set.num_users
        n_items = train_set.num_items
        n_aspects = sentiment_modality.num_aspects

        X_row = []
        X_col = []
        X_data = []
        Y_row = []
        Y_col = []
        Y_data = []
        for uid, isid in tqdm(
            sentiment_modality.user_sentiment.items(),
            disable=not self.verbose,
            desc="Building matrices",
        ):
            for iid, sid in isid.items():
                aos = sentiment_modality.sentiment[sid]
                aids = set(aid for aid, _, _ in aos)  # Only one per review/sid
                for aid in aids:
                    X_row.append(iid)
                    X_col.append(aid)
                    X_data.append(1)
                    Y_row.append(uid)
                    Y_col.append(aid)
                    Y_data.append(1)

        # Algorithm 1: Offline training line 2
        X = csr_matrix((X_data, (X_row, X_col)), shape=(n_items, n_aspects))
        Y = csr_matrix((Y_data, (Y_row, Y_col)), shape=(n_users, n_aspects))

        # Algorithm 1: Offline training line 3
        X.data = np.log2(X.data) + 1
        Y.data = np.log2(Y.data) + 1

        # Algorithm 1: Offline training line 4
        if self.verbose:
            print("Building symmetric normalized matrices R, X, Y")
        self.R = self._symmetrical_normalization(train_set.csr_matrix)
        self.X = self._symmetrical_normalization(X)
        self.Y = self._symmetrical_normalization(Y)

        if self.verbose:
            total_time = time() - start_time
            print("Building matrices completed in %d s" % total_time)

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
        self._init(train_set)

        if not self.trainable:
            return self

        # Offline training: Build item-aspect matrix X and user-aspect matrix Y
        self._create_matrices(train_set)
        return self

    def _online_recommendation(self, user):
        # Algorithm 1: Online recommendation line 5
        p_0 = self.r_mat[[user]]
        p_0.data.fill(1)
        p_0 = p_0.toarray().squeeze()
        a_0 = self.Y[user].toarray().squeeze()
        u_0 = np.zeros(self.r_mat.shape[0])
        u_0[user] = 1

        # Algorithm 1: Online training line 6
        if p_0.any():
            p_0 /= np.linalg.norm(p_0, 1)
        if a_0.any():
            a_0 /= np.linalg.norm(a_0, 1)
        if u_0.any():
            u_0 /= np.linalg.norm(u_0, 1)

        # Algorithm 1: Online recommendation line 7
        p = self.p.copy()
        a = self.a.copy()
        u = self.u.copy()

        # Algorithm 1: Online recommendation line 8
        prev_p = p
        prev_a = a
        prev_u = u
        inc = 1
        while True:
            # eq. 4
            u_denominator = self.alpha + self.gamma + self.eta_U + EPS
            u = (
                self.alpha / u_denominator * self.R * p
                + self.gamma / u_denominator * self.Y * a
                + self.eta_U / u_denominator * u_0
            ).squeeze()
            p_denominator = self.alpha + self.beta + self.eta_P + EPS
            p = (
                self.alpha / p_denominator * self.R.T * u
                + self.beta / p_denominator * self.X * a
                + self.eta_P / p_denominator * p_0
            ).squeeze()
            a_denominator = self.gamma + self.beta + self.eta_A + EPS
            a = (
                self.gamma / a_denominator * self.Y.T * u
                + self.beta / a_denominator * self.X.T * p
                + self.eta_P / a_denominator * a_0
            ).squeeze()

            if (self.max_iter > 0 and inc > self.max_iter) or (
                np.all(np.isclose(u, prev_u))
                and np.all(np.isclose(p, prev_p))
                and np.all(np.isclose(a, prev_a))
            ):  # stop when converged
                break
            prev_p, prev_a, prev_u = p, a, u
            inc += 1

        # Algorithm 1: Online recommendation line 9
        return p, a, u

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
        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)

        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        item_scores, *_ = self._online_recommendation(user_idx)
        # Set already rated items to zero.
        item_scores[self.r_mat[user_idx].indices] = 0

        # Scale to match rating scale.
        item_scores = (
            item_scores * (self.max_rating - self.min_rating) / max(item_scores)
            + self.min_rating
        )

        return item_scores if item_idx is None else item_scores[item_idx]
