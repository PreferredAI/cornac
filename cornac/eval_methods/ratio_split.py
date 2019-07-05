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

from math import ceil

import numpy as np

from .base_method import BaseMethod
from ..utils import get_rng
from ..utils.common import safe_indexing


class RatioSplit(BaseMethod):
    """Train-Test Split Evaluation Method.

    Parameters
    ----------

    data: ..., required
        The input data in the form of triplets (user, item, rating).

    fmt: str, optional, default: "UIR"
        The format of input data:
        - UIR: (user, item, rating) triplet data
        - UIRT: (user, item , rating, timestamp) quadruplet data

    test_size: float, optional, default: 0.2
        The proportion of the test set, \
        if > 1 then it is treated as the size of the test set.

    val_size: float, optional, default: 0.0
        The proportion of the validation set, \
        if > 1 then it is treated as the size of the validation set.

    rating_threshold: float, optional, default: 1.
        The minimum value that is considered to be a good rating used for ranking, \
        e.g, if the ratings are in {1, ..., 5}, then rating_threshold = 4.

    shuffle: bool, optional, default: True
        Shuffle the data before splitting.

    seed: int, optional, default: None
        Random seed for reproduce the splitting.

    exclude_unknowns: bool, optional, default: False
        Ignore unknown users and items (cold-start) during evaluation and testing

    verbose: bool, optional, default: False
        Output running log
    """

    def __init__(self, data, fmt='UIR', test_size=0.2, val_size=0.0, rating_threshold=1.0, shuffle=True,
                 seed=None, exclude_unknowns=False, verbose=False, **kwargs):
        BaseMethod.__init__(self, data=data, fmt=fmt, rating_threshold=rating_threshold,
                            exclude_unknowns=exclude_unknowns, verbose=verbose, **kwargs)
        self._shuffle = shuffle
        self._seed = seed
        self._train_size, self._val_size, self._test_size = self.validate_size(val_size, test_size, len(self._data))
        self._split()

    @staticmethod
    def validate_size(val_size, test_size, num_ratings):
        if val_size is None:
            val_size = 0.0
        elif val_size < 0:
            raise ValueError('val_size={} should be greater than zero'.format(val_size))
        elif val_size >= num_ratings:
            raise ValueError(
                'val_size={} should be less than the number of ratings {}'.format(val_size, num_ratings))

        if test_size is None:
            test_size = 0.0
        elif test_size < 0:
            raise ValueError('test_size={} should be greater than zero'.format(test_size))
        elif test_size >= num_ratings:
            raise ValueError(
                'test_size={} should be less than the number of ratings {}'.format(test_size, num_ratings))

        if val_size < 1:
            val_size = ceil(val_size * num_ratings)
        if test_size < 1:
            test_size = ceil(test_size * num_ratings)

        if val_size + test_size >= num_ratings:
            raise ValueError(
                'The sum of val_size and test_size ({}) should be smaller than the number of ratings {}'.format(
                    val_size + test_size, num_ratings))

        train_size = num_ratings - (val_size + test_size)

        return int(train_size), int(val_size), int(test_size)

    def _split(self):
        if self.verbose:
            print("Splitting the data")

        data_idx = np.arange(len(self._data))
        if self._shuffle:
            data_idx = get_rng(self._seed).permutation(data_idx)

        train_idx = data_idx[:self._train_size]
        test_idx = data_idx[-self._test_size:]
        val_idx = data_idx[self._train_size:-self._test_size]

        train_data = safe_indexing(self._data, train_idx)
        test_data = safe_indexing(self._data, test_idx)
        val_data = safe_indexing(self._data, val_idx) if len(val_idx) > 0 else None

        self.build(train_data=train_data, test_data=test_data, val_data=val_data)
        self._split_ran = True

        if self.verbose:
            print('Total users = {}'.format(self.total_users))
            print('Total items = {}'.format(self.total_items))
