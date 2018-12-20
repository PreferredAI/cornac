# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..utils.generic_utils import safe_indexing, validate_data_format
from math import ceil
from .base_strategy import BaseStrategy
from ..data import MatrixTrainSet, TestSet
import numpy as np


class RatioSplit(BaseStrategy):
    """Train-Test Split Evaluation Strategy.

    Parameters
    ----------

    data: ..., required
        The input data in the form of triplets (user, item, rating).

    data_format: str, optional, default: "UIR"
        The format of input data:
        - UIR: (user, item, rating) triplet data
        - UIRT: (user, item , rating, timestamp) quadruplet data

    val_size: float, optional, default: 0.0
        The proportion of the validation set, \
        if > 1 then it is treated as the size of the validation set.

    test_size: float, optional, default: 0.2
        The proportion of the test set, \
        if > 1 then it is treated as the size of the test set.

    rating_threshold: float, optional, default: 1.
        The minimum value that is considered to be a good rating used for ranking, \
        e.g, if the ratings are in {1, ..., 5}, then rating_threshold = 4.

    shuffle: bool, optional, default: True
        Shuffle the data before splitting.

    exclude_unknowns: bool, optional, default: False
        Ignore unknown users and items (cold-start) during evaluation and testing

    verbose: bool, optional, default: False
        Output running log
    """

    def __init__(self, data, data_format='UIR', val_size=0.0, test_size=0.2, rating_threshold=1., shuffle=True,
                 random_state=None, exclude_unknowns=False, verbose=False):
        BaseStrategy.__init__(self, data=data, data_format=data_format, rating_threshold=rating_threshold,
                              exclude_unknowns=exclude_unknowns, verbose=verbose)

        self._shuffle = shuffle
        self._random_state = random_state
        self._train_size, self._val_size, self._test_size = self._validate_sizes(val_size, test_size, len(self._data))
        self._split_ran = False


    @staticmethod
    def _validate_sizes(val_size, test_size, num_ratings):
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


    def split(self):
        if self._split_ran:
            if self.verbose:
                print('Data is already split!')
            return

        if self.verbose:
            print("Splitting the data")

        data_idx = np.arange(len(self._data))

        if self._shuffle:
            if not self._random_state is None:
                np.random.set_state(self._random_state)
            data_idx = np.random.permutation(data_idx)

        train_idx = data_idx[:self._train_size]
        val_idx = data_idx[self._train_size:(self._train_size + self._val_size)]
        test_idx = data_idx[-self._test_size:]

        train_data = safe_indexing(self._data, train_idx)
        val_data = safe_indexing(self._data, val_idx)
        test_data = safe_indexing(self._data, test_idx)

        if self._data_format == 'UIR':
            self.build_from_uir_format(train_data, val_data, test_data)

        self._split_ran = True

        if self.verbose:
            print('Total users = {}'.format(self.total_users))
            print('Total items = {}'.format(self.total_items))


    def evaluate(self, model, metrics, user_based):
        self.split()
        return BaseStrategy.evaluate(self, model, metrics, user_based)
