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

from .base_method import BaseMethod
from ..utils import get_rng
from ..utils.common import safe_indexing


class ChronoRatioSplit(BaseMethod):
    """Splitting data into training, validation, and test sets chronologically.
    Validation and test data is always shuffled before split.

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value, review_time)].

    test_size: float, optional, default: 0.2
        The proportion of the test set.

    val_size: float, optional, default: 0.0
        The proportion of the validation set.

    rating_threshold: float, optional, default: 1.0
        Threshold used to binarize rating values into positive or negative feedback for
        model evaluation using ranking metrics (rating metrics are not affected).

    seed: int, optional, default: None
        Random seed for reproducibility.

    exclude_unknowns: bool, optional, default: True
        If `True`, unknown users and items will be ignored during model evaluation.

    verbose: bool, optional, default: False
        Output running log.

    """

    def __init__(self, data, fmt='UIRT', test_size=0.2, val_size=0.0, rating_threshold=1.0,
                 seed=None, exclude_unknowns=True, verbose=False, **kwargs):
        super().__init__(data=data, fmt=fmt, rating_threshold=rating_threshold, seed=seed,
                         exclude_unknowns=exclude_unknowns, verbose=verbose, **kwargs)
        if fmt != 'UIRT' or len(self._data[0]) != 4:
            raise ValueError('Input data must be in "UIRT" format')
        self.train_size, self.val_size, self.test_size = self.validate_size(val_size, test_size)
        self._split()

    @staticmethod
    def validate_size(val_size, test_size):
        if val_size is None:
            val_size = 0.0
        elif val_size < 0 or val_size > 1.0:
            raise ValueError('val_size={} should be within range [0, 1]'.format(val_size))

        if test_size is None:
            test_size = 0.0
        elif test_size < 0 or test_size > 1.0:
            raise ValueError('test_size={} should be within range [0, 1]'.format(test_size))

        train_size = 1. - val_size - test_size

        if train_size < 0:
            raise ValueError('The total sum of val_size={} and test_size={} should be less than 1.0'.format(
                val_size, test_size))

        return train_size, val_size, test_size

    def _split(self):
        sorted_data = sorted(self._data, key=lambda x: x[3])

        user_data = {}
        for idx, tup in enumerate(sorted_data):
            user_data.setdefault(tup[0], []).append(idx)

        train_idx = []
        test_idx = []
        val_idx = []
        for items in user_data.values():
            n_ratings = len(items)
            n_test = int(self.test_size * n_ratings)
            n_val = int(self.val_size * n_ratings)
            n_train = n_ratings - n_test - n_val

            non_training_idx = self.rng.permutation(items[n_train:]).tolist()
            train_idx += items[:n_train]
            test_idx += non_training_idx[-n_test:]
            val_idx += non_training_idx[:-n_test]

        train_data = safe_indexing(sorted_data, train_idx)
        test_data = safe_indexing(sorted_data, test_idx)
        val_data = safe_indexing(sorted_data, val_idx) if len(val_idx) > 0 else None
        self.build(train_data=train_data, test_data=test_data, val_data=val_data)
