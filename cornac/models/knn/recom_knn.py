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

import multiprocessing

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.fast_sparse_funcs import inplace_csr_row_normalize_l2
from .similarity import compute_similarity, compute_score, compute_score_single


EPS = 1e-8

SIMILARITIES = ["cosine", "pearson"]
WEIGHTING_OPTIONS = ["idf", "bm25"]


def _mean_centered(ui_mat):
    """Subtract every rating values with mean value of the corresponding rows"""
    mean_arr = np.zeros(ui_mat.shape[0])
    for i in range(ui_mat.shape[0]):
        start_idx, end_idx = ui_mat.indptr[i : i + 2]
        mean_arr[i] = np.mean(ui_mat.data[start_idx:end_idx])
        row_data = ui_mat.data[start_idx:end_idx]
        row_data -= mean_arr[i]
        row_data[row_data == 0] = EPS
        ui_mat.data[start_idx:end_idx] = row_data

    return ui_mat, mean_arr


def _amplify(ui_mat, alpha=1.0):
    """Exponentially amplify values of similarity matrix"""
    if alpha == 1.0:
        return ui_mat

    for i, w in enumerate(ui_mat.data):
        ui_mat.data[i] = w ** alpha if w > 0 else -(-w) ** alpha
    return ui_mat


def _idf_weight(ui_mat):
    """Weight the matrix Inverse Document (Item) Frequency"""
    X = coo_matrix(ui_mat)

    # calculate IDF
    N = float(X.shape[0])
    idf = np.log(N / np.bincount(X.col))

    weights = idf[ui_mat.indices] + EPS
    return weights


def _bm25_weight(ui_mat):
    """Weight the matrix with BM25 algorithm"""
    K1 = 1.2
    B = 0.8

    X = coo_matrix(ui_mat)
    X.data = np.ones_like(X.data)

    N = float(X.shape[0])
    idf = np.log(N / np.bincount(X.col))

    # calculate length_norm per document (user)
    row_sums = np.ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # bm25 weights
    weights = (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col] + EPS
    return weights


class UserKNN(Recommender):
    """User-Based Nearest Neighbor.

    Parameters
    ----------
    name: string, default: 'UserKNN'
        The name of the recommender model.

    k: int, optional, default: 20
        The number of nearest neighbors.
       
    similarity: str, optional, default: 'cosine'
        The similarity measurement. Supported types: ['cosine', 'pearson']
    
    mean_centered: bool, optional, default: False
        Whether values of the user-item rating matrix will be centered by the mean
        of their corresponding rows (mean rating of each user).  
    
    weighting: str, optional, default: None
        The option for re-weighting the rating matrix. Supported types: ['idf', 'bm25'].
        If None, no weighting is applied.
          
    amplify: float, optional, default: 1.0
        Amplifying the influence on similarity weights.
        
    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    * CarlKadie, J. B. D. (1998). Empirical analysis of predictive algorithms for collaborative filtering. Microsoft Research Microsoft Corporation One Microsoft Way Redmond, WA, 98052.
    * Aggarwal, C. C. (2016). Recommender systems (Vol. 1). Cham: Springer International Publishing.
    """

    def __init__(
        self,
        name="UserKNN",
        k=20,
        similarity="cosine",
        mean_centered=False,
        weighting=None,
        amplify=1.0,
        num_threads=0,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.similarity = similarity
        self.mean_centered = mean_centered
        self.weighting = weighting
        self.amplify = amplify
        self.seed = seed
        self.rng = get_rng(seed)

        if self.similarity not in SIMILARITIES:
            raise ValueError(
                "Invalid similarity choice, supported {}".format(SIMILARITIES)
            )

        if self.weighting is not None and self.weighting not in WEIGHTING_OPTIONS:
            raise ValueError(
                "Invalid weighting choice, supported {}".format(WEIGHTING_OPTIONS)
            )

        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

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

        self.ui_mat = self.train_set.matrix.copy()
        self.mean_arr = np.zeros(self.ui_mat.shape[0])
        if self.train_set.min_rating != self.train_set.max_rating:  # explicit feedback
            self.ui_mat, self.mean_arr = _mean_centered(self.ui_mat)

        if self.mean_centered or self.similarity == "pearson":
            weight_mat = self.ui_mat.copy()
        else:
            weight_mat = self.train_set.matrix.copy()

        # re-weighting
        if self.weighting == "idf":
            weight_mat.data *= np.sqrt(_idf_weight(self.train_set.matrix))
        elif self.weighting == "bm25":
            weight_mat.data *= np.sqrt(_bm25_weight(self.train_set.matrix))

        # only need item-user matrix for prediction
        self.iu_mat = self.ui_mat.T.tocsr()
        del self.ui_mat

        self.sim_mat = compute_similarity(
            weight_mat, k=self.k, num_threads=self.num_threads, verbose=self.verbose
        )
        self.sim_mat = _amplify(self.sim_mat, self.amplify)

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
        if self.train_set.is_unk_user(user_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d)" % user_idx
            )

        if item_idx is not None and self.train_set.is_unk_item(item_idx):
            raise ScoreException(
                "Can't make score prediction for (item_id=%d)" % item_idx
            )

        if item_idx is not None:
            weighted_avg = compute_score_single(
                True,
                self.sim_mat[user_idx].A.ravel(),
                self.iu_mat.indptr[item_idx],
                self.iu_mat.indptr[item_idx + 1],
                self.iu_mat.indices,
                self.iu_mat.data,
                k=self.k,
            )
            return self.mean_arr[user_idx] + weighted_avg

        weighted_avg = np.zeros(self.train_set.num_items)
        compute_score(
            True,
            self.sim_mat[user_idx].A.ravel(),
            self.iu_mat.indptr,
            self.iu_mat.indices,
            self.iu_mat.data,
            k=self.k,
            num_threads=self.num_threads,
            output=weighted_avg,
        )
        known_item_scores = self.mean_arr[user_idx] + weighted_avg

        return known_item_scores


class ItemKNN(Recommender):
    """Item-Based Nearest Neighbor.

    Parameters
    ----------
    name: string, default: 'ItemKNN'
        The name of the recommender model.

    k: int, optional, default: 20
        The number of nearest neighbors.

    similarity: str, optional, default: 'cosine'
        The similarity measurement. Supported types: ['cosine', 'pearson']
      
    mean_centered: bool, optional, default: False
        Whether values of the user-item rating matrix will be centered by the mean
        of their corresponding rows (mean rating of each user).  
         
    weighting: str, optional, default: None
        The option for re-weighting the rating matrix. Supported types: ['idf', 'bm25'].
        If None, no weighting is applied.
               
    amplify: float, optional, default: 1.0
        Amplifying the influence on similarity weights.
         
    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    * Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001, April). Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th international conference on World Wide Web (pp. 285-295).
    * Aggarwal, C. C. (2016). Recommender systems (Vol. 1). Cham: Springer International Publishing.
    """

    def __init__(
        self,
        name="ItemKNN",
        k=20,
        similarity="cosine",
        mean_centered=False,
        weighting=None,
        amplify=1.0,
        num_threads=0,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.similarity = similarity
        self.mean_centered = mean_centered
        self.weighting = weighting
        self.amplify = amplify
        self.seed = seed
        self.rng = get_rng(seed)

        if self.similarity not in SIMILARITIES:
            raise ValueError(
                "Invalid similarity choice, supported {}".format(SIMILARITIES)
            )

        if self.weighting is not None and self.weighting not in WEIGHTING_OPTIONS:
            raise ValueError(
                "Invalid weighting choice, supported {}".format(WEIGHTING_OPTIONS)
            )

        if seed is not None:
            self.num_threads = 1
        elif num_threads > 0 and num_threads < multiprocessing.cpu_count():
            self.num_threads = num_threads
        else:
            self.num_threads = multiprocessing.cpu_count()

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

        self.ui_mat = self.train_set.matrix.copy()
        self.mean_arr = np.zeros(self.ui_mat.shape[0])
        if self.train_set.min_rating != self.train_set.max_rating:  # explicit feedback
            self.ui_mat, self.mean_arr = _mean_centered(self.ui_mat)

        if self.mean_centered:
            weight_mat = self.ui_mat.copy()
        else:
            weight_mat = self.train_set.matrix.copy()

        if self.similarity == "pearson":  # centered by columns
            weight_mat, _ = _mean_centered(weight_mat.T.tocsr())
            weight_mat = weight_mat.T.tocsr()

        # re-weighting
        if self.weighting == "idf":
            weight_mat.data *= np.sqrt(_idf_weight(self.train_set.matrix))
        elif self.weighting == "bm25":
            weight_mat.data *= np.sqrt(_bm25_weight(self.train_set.matrix))

        weight_mat = weight_mat.T.tocsr()
        self.sim_mat = compute_similarity(
            weight_mat, k=self.k, num_threads=self.num_threads, verbose=self.verbose
        )
        self.sim_mat = _amplify(self.sim_mat, self.amplify)

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
        if self.train_set.is_unk_user(user_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d)" % user_idx
            )

        if item_idx is not None and self.train_set.is_unk_item(item_idx):
            raise ScoreException(
                "Can't make score prediction for (item_id=%d)" % item_idx
            )

        if item_idx is not None:
            weighted_avg = compute_score_single(
                False,
                self.ui_mat[user_idx].A.ravel(),
                self.sim_mat.indptr[item_idx],
                self.sim_mat.indptr[item_idx + 1],
                self.sim_mat.indices,
                self.sim_mat.data,
                k=self.k,
            )
            return self.mean_arr[user_idx] + weighted_avg

        weighted_avg = np.zeros(self.train_set.num_items)
        compute_score(
            False,
            self.ui_mat[user_idx].A.ravel(),
            self.sim_mat.indptr,
            self.sim_mat.indices,
            self.sim_mat.data,
            k=self.k,
            num_threads=self.num_threads,
            output=weighted_avg,
        )
        return self.mean_arr[user_idx] + weighted_avg
