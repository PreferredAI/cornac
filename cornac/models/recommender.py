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

from ..exception import ScoreException
from ..utils.common import intersects, clip


class Recommender:
    """Generic class for a recommender model. All recommendation models should inherit from this class 
    
    Parameters
    ----------------
    name: str, required
        The name of the recommender model

    trainable: boolean, optional, default: True
        When False, the model is not trainable

    """

    def __init__(self, name, trainable=True, verbose=False):
        self.name = name
        self.trainable = trainable
        self.verbose = verbose
        self.train_set = None
        self.val_set = None

        self.best_value = -np.Inf
        self.best_epoch = 0
        self.current_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def fit(self, train_set, val_set=None):
        """Fit the model to observations. Need to

        Parameters
        ----------
        train_set: object of type TrainSet, required
            An object containing the user-item preference in csr scipy sparse format,\
            as well as some useful attributes such as mappings to the original user/item ids.\
            Please refer to the class TrainSet in the "data" module for details.

        val_set: object of type TestSet, optional, default: None
            An object containing the user-item preference for model selection purposes (e.g., early stopping).
            Please refer to the class TestSet in the "data" module for details.

        Returns
        -------
        self : object
        """
        self.train_set = train_set
        self.val_set = val_set
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
        raise NotImplementedError('The algorithm is not able to make score prediction!')

    def default_score(self):
        """Overwrite this function if your algorithm has special treatment for cold-start problem

        """
        return self.train_set.global_mean

    def rate(self, user_idx, item_idx, clipping=True):
        """Give a rating score between pair of user and item

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform item raking.

        item_idx: int, required
            The index of the item to be rated by the user.

        clipping: bool, default: True
            Whether to clip the predicted rating value.

        Returns
        -------
        A scalar
            A rating score of the user for the item
        """
        try:
            rating_pred = self.score(user_idx, item_idx)
        except ScoreException:
            rating_pred = self.default_score()

        if clipping:
            rating_pred = clip(values=rating_pred,
                               lower_bound=self.train_set.min_rating,
                               upper_bound=self.train_set.max_rating)

        return rating_pred

    def rank(self, user_idx, item_indices=None):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform item raking.

        item_indices: 1d array, optional, default: None
            A list of candidate item indices to be ranked by the user.
            If `None`, list of ranked known item indices and their scores will be returned

        Returns
        -------
        Tuple of `item_rank`, and `item_scores`. The order of values
        in item_scores are corresponding to the order of their ids in item_ids

        """
        try:
            known_item_scores = self.score(user_idx)
        except ScoreException:
            known_item_scores = np.ones(self.train_set.num_items) * self.default_score()

        if item_indices is None:
            item_scores = known_item_scores
            item_rank = item_scores.argsort()[::-1]
        else:
            num_items = max(self.train_set.num_items, max(item_indices) + 1)
            item_scores = np.ones(num_items) * np.min(known_item_scores)
            item_scores[:self.train_set.num_items] = known_item_scores
            item_rank = item_scores.argsort()[::-1]
            item_rank = intersects(item_rank, item_indices, assume_unique=True)
            item_scores = item_scores[item_indices]
        return item_rank, item_scores

    def monitor_value(self):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.
        Note: `val_set` could be `None` thus it needs to be checked before usage.

        Returns
        -------
        :raise NotImplementedError
        """
        raise NotImplementedError()

    def early_stop(self, min_delta=0., patience=0):
        """Check if training should be stopped when validation loss has stopped improving.

        Parameters
        ----------
        min_delta: float, optional, default: 0.
            The minimum increase in monitored value on validation set to be considered as improvement,
            i.e. an increment of less than `min_delta` will count as no improvement.

        patience: int, optional, default: 0
            Number of epochs with no improvement after which training should be stopped.

        Returns
        -------
        res : bool
            Return `True` if model training should be stopped (no improvement on validation set),
            otherwise return `False`.
        """
        self.current_epoch += 1
        current_value = self.monitor_value()
        if current_value is None:
            return False

        if np.greater_equal(current_value - self.best_value, min_delta):
            self.best_value = current_value
            self.best_epoch = self.current_epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= patience:
                self.stopped_epoch = self.current_epoch

        if self.stopped_epoch > 0:
            print('Early stopping:')
            print('- best epoch = {}, stopped epoch = {}'.format(self.best_epoch, self.stopped_epoch))
            print('- best monitored value = {:.6f} (delta = {:.6f})'.format(
                self.best_value, current_value - self.best_value))
            return True
        return False