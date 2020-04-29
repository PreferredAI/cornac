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
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.fast_sparse_funcs import inplace_csr_row_normalize_l2

from .similarity import compute_similarity


EPS = 1e-8


def _mean_centered(csr_mat):
    """Subtract every rating values with mean value of the corresponding rows"""
    mean_arr = np.zeros(csr_mat.shape[0])
    for i in range(csr_mat.shape[0]):
        start_idx, end_idx = csr_mat.indptr[i : i + 2]
        mean_arr[i] = np.mean(csr_mat.data[start_idx:end_idx])
        csr_mat.data[start_idx:end_idx] -= mean_arr[i]

    return csr_mat, mean_arr


def _tfidf_weight(csr_mat):
    """Weight the matrix with TF-IDF"""
    # calculate IDF
    N = float(csr_mat.shape[1])
    idf = np.log(N) - np.log1p(np.bincount(csr_mat.indices))

    # apply TF-IDF adjustment
    csr_mat.data *= np.sqrt(idf[csr_mat.indices])
    return csr_mat


def _bm25_weight(csr_mat):
    """Weight the matrix with BM25 algorithm"""
    K1 = 1.2
    B = 0.8

    # calculate IDF
    N = float(csr_mat.shape[1])
    idf = np.log(N) - np.log1p(np.bincount(csr_mat.indices))

    # calculate length_norm per document
    row_sums = np.ravel(csr_mat.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by BM25
    row_counts = np.ediff1d(csr_mat.indptr)
    row_inds = np.repeat(np.arange(csr_mat.shape[0]), row_counts)
    weights = (
        (K1 + 1.0) / (K1 * length_norm[row_inds] + csr_mat.data) * idf[csr_mat.indices]
    )
    csr_mat.data *= np.sqrt(weights)
    return csr_mat


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
    
    weighting: str, optional, default: None
        The option for re-weighting the rating matrix. Supported types: [tf-idf', 'bm25'].
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

    SIMILARITIES = ["cosine", "pearson"]
    WEIGHTING_OPTIONS = ["tf-idf", "bm25"]

    def __init__(
        self,
        name="UserKNN",
        k=20,
        similarity="cosine",
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
        self.weighting = weighting
        self.amplify = amplify
        self.seed = seed
        self.rng = get_rng(seed)

        if self.similarity not in self.SIMILARITIES:
            raise ValueError(
                "Invalid similarity choice, supported {}".format(self.SIMILARITIES)
            )

        if self.weighting is not None and self.weighting not in self.WEIGHTING_OPTIONS:
            raise ValueError(
                "Invalid weighting choice, supported {}".format(self.WEIGHTING_OPTIONS)
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

        if self.similarity == "cosine":
            weight_mat = self.train_set.matrix.copy()
        elif self.similarity == "pearson":
            weight_mat = self.ui_mat.copy()

        # rating matrix re-weighting
        if self.weighting == "tf-idf":
            weight_mat = _tfidf_weight(weight_mat)
        elif self.weighting == "bm25":
            weight_mat = _bm25_weight(weight_mat)

        inplace_csr_row_normalize_l2(weight_mat)
        self.sim_mat = compute_similarity(
            weight_mat, k=self.k, num_threads=self.num_threads, verbose=self.verbose
        ).power(self.amplify)

        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
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

        user_weights = self.sim_mat[user_idx]
        user_weights = user_weights / (
            np.abs(user_weights).sum() + EPS
        )  # normalize for rating prediction
        known_item_scores = (
            self.mean_arr[user_idx] + user_weights.dot(self.ui_mat).A.ravel()
        )

        if item_idx is not None:
            return known_item_scores[item_idx]

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
        The similarity measurement. Supported types: ['cosine', 'adjusted', 'pearson']
       
    weighting: str, optional, default: None
        The option for re-weighting the rating matrix. Supported types: [tf-idf', 'bm25'].
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

    SIMILARITIES = ["cosine", "adjusted", "pearson"]
    WEIGHTING_OPTIONS = ["tf-idf", "bm25"]

    def __init__(
        self,
        name="ItemKNN",
        k=20,
        similarity="cosine",
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
        self.weighting = weighting
        self.amplify = amplify
        self.seed = seed
        self.rng = get_rng(seed)

        if self.similarity not in self.SIMILARITIES:
            raise ValueError(
                "Invalid similarity choice, supported {}".format(self.SIMILARITIES)
            )

        if self.weighting is not None and self.weighting not in self.WEIGHTING_OPTIONS:
            raise ValueError(
                "Invalid weighting choice, supported {}".format(self.WEIGHTING_OPTIONS)
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

        explicit_feedback = self.train_set.min_rating != self.train_set.max_rating
        if explicit_feedback: 
            self.ui_mat, self.mean_arr = _mean_centered(self.ui_mat)

        if self.similarity == "cosine":
            weight_mat = self.train_set.matrix.copy()
        elif self.similarity == "adjusted":
            weight_mat = self.ui_mat.copy()  # mean-centered by rows
        elif self.similarity == "pearson" and explicit_feedback:
            weight_mat, _ = _mean_centered(
                self.train_set.matrix.T.tocsr()
            )  # mean-centered by columns
            weight_mat = weight_mat.T.tocsr()

        # rating matrix re-weighting
        if self.weighting == "tf-idf":
            weight_mat = _tfidf_weight(weight_mat)
        elif self.weighting == "bm25":
            weight_mat = _bm25_weight(weight_mat)

        weight_mat = weight_mat.T.tocsr()
        inplace_csr_row_normalize_l2(weight_mat)
        self.sim_mat = compute_similarity(
            weight_mat, k=self.k, num_threads=self.num_threads, verbose=self.verbose
        ).power(self.amplify)

        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
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

        user_profile = self.ui_mat[user_idx]
        known_item_scores = self.mean_arr[user_idx] + (
            user_profile.dot(self.sim_mat).A.ravel()
            / (np.abs(self.sim_mat).sum(axis=0).A.ravel() + EPS)
        )

        if item_idx is not None:
            return known_item_scores[item_idx]

        return known_item_scores
