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

from .base_method import BaseMethod
from ..utils.common import safe_indexing
from ..experiment.result import CVResult
from ..utils import get_rng


class CrossValidation(BaseMethod):
    """Cross Validation Evaluation Method.

    Parameters
    ----------
    data: ... , required
        Input data in the triplet format (user_id, item_id, rating_val).

    n_folds: int, optional, default: 5
        The number of folds for cross validation.

    rating_threshold: float, optional, default: 1.
        The minimum value that is considered to be a good rating, \
        e.g, if the ratings are in {1, ... ,5}, then rating_threshold = 4.

    partition: array-like, shape (n_observed_ratings,), optional, default: None
        The partition of ratings into n_folds (fold label of each rating) \
        If None, random partitioning is performed to assign each rating into a fold.

    rating_threshold: float, optional, default: 1.
        The minimum value that is considered to be a good rating used for ranking, \
        e.g, if the ratings are in {1, ..., 5}, then rating_threshold = 4.

    seed: int, optional, default: None
        Random seed for reproduce the splitting.

    exclude_unknowns: bool, optional, default: False
        Ignore unknown users and items (cold-start) during evaluation and testing

    verbose: bool, optional, default: False
        Output running log
    """

    def __init__(self, data, fmt='UIR', n_folds=5, rating_threshold=1., partition=None,
                 seed=None, exclude_unknowns=True, verbose=False, **kwargs):
        BaseMethod.__init__(self, data=data, fmt=fmt, rating_threshold=rating_threshold,
                            exclude_unknowns=exclude_unknowns, verbose=verbose, **kwargs)
        self.n_folds = n_folds
        self.n_ratings = len(self._data)
        self.current_fold = 0
        self.current_split = None

        self._seed = seed
        self._partition = self._validate_partition(partition)

    # Partition ratings into n_folds
    def _partition_data(self):
        rng = get_rng(self._seed)

        fold_size = int(self.n_ratings / self.n_folds)
        remain_size = self.n_ratings - fold_size * self.n_folds

        partition = np.repeat(np.arange(self.n_folds), fold_size)
        rng.shuffle(partition)

        if remain_size > 0:
            remain_partition = rng.choice(self.n_folds, size=remain_size, replace=True, p=None)
            partition = np.concatenate((partition, remain_partition))

        return partition

    def _validate_partition(self, partition):
        if partition is None:
            return self._partition_data()
        elif len(partition) != self.n_ratings:
            raise ValueError('The partition length must be equal to the number of ratings')
        elif len(set(partition)) != self.n_folds:
            raise ValueError('Number of folds in given partition different from %s' % (self.n_folds))

        return partition

    def _get_train_test(self):
        if self.verbose:
            print('Fold: {}'.format(self.current_fold + 1))

        test_idx = np.where(self._partition == self.current_fold)[0]
        train_idx = np.where(self._partition != self.current_fold)[0]

        train_data = safe_indexing(self._data, train_idx)
        test_data = safe_indexing(self._data, test_idx)
        self.build(train_data=train_data, test_data=test_data)

        if self.verbose:
            print('Total users = {}'.format(self.total_users))
            print('Total items = {}'.format(self.total_items))

    def _next_fold(self):
        if self.current_fold < self.n_folds - 1:
            self.current_fold = self.current_fold + 1
        else:
            self.current_fold = 0

    def evaluate(self, model, metrics, user_based):
        result = CVResult(model.name)
        for fold in range(self.n_folds):
            self._get_train_test()
            fold_result = BaseMethod.evaluate(self, model, metrics, user_based)
            result.append(fold_result)
            self._next_fold()
        result.organize()
        return result
