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
from scipy.sparse import csr_matrix
from tqdm import trange

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.fast_sparse_funcs import inplace_csr_row_normalize_l2

from .similarity import compute_similarity


EPS = 1e-8


def _mean_centered(csr_mat):
    mean_arr = np.zeros(csr_mat.shape[0])
    for i in range(csr_mat.shape[0]):
        start_idx, end_idx = csr_mat.indptr[i : i + 2]
        mean_arr[i] = np.mean(csr_mat.data[start_idx:end_idx])
        csr_mat.data[start_idx:end_idx] -= mean_arr[i]

    return csr_mat, mean_arr


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
        
    num_threads: int, optional, default: 0
        Number of parallel threads for training. If num_threads=0, all CPU cores will be utilized.
        If seed is not None, num_threads=1 to remove randomness from parallelization.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    
    """

    SIMILARITIES = ["cosine", "pearson"]

    def __init__(
        self,
        name="UserKNN",
        k=20,
        similarity="cosine",
        num_threads=0,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.similarity = similarity
        self.seed = seed
        self.rng = get_rng(seed)

        if self.similarity not in self.SIMILARITIES:
            raise ValueError(
                "Invalid similarity choice, supported {}".format(self.SIMILARITIES)
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

        if self.similarity == "pearson":
            self.ui_mat, self.mean_arr = _mean_centered(self.ui_mat)

        normalized_ui_mat = self.ui_mat.copy()
        inplace_csr_row_normalize_l2(normalized_ui_mat)
        self.sim_mat = compute_similarity(
            normalized_ui_mat,
            k=self.k,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )

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
            user_weights.sum() + EPS
        )  # normalize for rating prediction
        known_item_scores = (
            self.mean_arr[user_idx] + user_weights.dot(self.ui_mat).A.ravel()
        )

        if item_idx is not None:
            return known_item_scores[item_idx]

        return known_item_scores
