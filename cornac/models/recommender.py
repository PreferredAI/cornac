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
from ..utils.common import intersects, excepts, clip


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

    def fit(self, train_set):
        """Fit the model with training data, should be called before each implementation of any recommender model's class

        """
        self.train_set = train_set

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
        raise NotImplementedError('The algorithm is not able to make score prediction!')

    def default_score(self):
        """Overwrite this function if your algorithm has special treatment for cold-start problem

        """
        return self.train_set.global_mean

    def default_rank(self, item_ids=None):
        """Overwrite this function if your algorithm has special treatment for cold-start problem

        """
        known_item_rank = self.train_set.item_ppl_rank
        known_item_scores = self.train_set.item_ppl_scores

        if item_ids is None:
            item_rank = known_item_rank
            item_scores = known_item_scores
        else:
            known_item_ids = intersects(known_item_rank, item_ids, assume_unique=True)
            unk_item_ids = excepts(known_item_rank, item_ids, assume_unique=True)
            item_rank = np.concatenate((known_item_ids, unk_item_ids))

            num_items = max(self.train_set.num_items, max(item_ids) + 1)
            item_scores = np.ones(num_items) * np.min(known_item_scores)
            item_scores[:self.train_set.num_items] = known_item_scores
            item_scores = item_scores[item_ids]

        return item_rank, item_scores

    def rate(self, user_id, item_id, clipping=True):
        """Give a rating score between pair of user and item

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform item raking.

        item_id: int, required
            The index of the item to be rated by the user.

        clipping: bool, default: True
            Whether to clip the predicted rating value.

        Returns
        -------
        A scalar
            A rating score of the user for the item
        """
        try:
            rating_pred = self.score(user_id, item_id)
        except ScoreException:
            rating_pred = self.default_score()

        if clipping:
            rating_pred = clip(values=rating_pred,
                               lower_bound=self.train_set.min_rating,
                               upper_bound=self.train_set.max_rating)

        return rating_pred

    def rank(self, user_id, item_ids=None):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform item raking.

        item_ids: 1d array, optional, default: None
            A list of candidate item indices to be ranked by the user.
            If `None`, list of ranked known item indices and their scores will be returned

        Returns
        -------
        Tuple of `item_rank`, and `item_scores`. The order of values
        in item_scores are corresponding to the order of their ids in item_ids

        """
        try:
            known_item_scores = self.score(user_id)
        except ScoreException:
            known_item_scores = np.ones(self.train_set.num_items) * self.default_score()

        if item_ids is None:
            item_scores = known_item_scores
            item_rank = item_scores.argsort()[::-1]
        else:
            num_items = max(self.train_set.num_items, max(item_ids) + 1)
            item_scores = np.ones(num_items) * np.min(known_item_scores)
            item_scores[:self.train_set.num_items] = known_item_scores
            item_rank = item_scores.argsort()[::-1]
            item_rank = intersects(item_rank, item_ids, assume_unique=True)
            item_scores = item_scores[item_ids]
        return item_rank, item_scores
