# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from math import ceil
from random import shuffle
from .base_strategy import BaseStrategy
from ..data import MatrixTrainSet, TestSet


class RatioSplit(BaseStrategy):

    """Train-Test Split Evaluation Strategy.

    Parameters
    ----------

    data: ..., required
        The input data in the form of triplets (user, item, rating).

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

    def __init__(self, data, val_size=0.0, test_size=0.2, rating_threshold=1., shuffle=True,
                 exclude_unknowns=False, verbose=False):
        super().__init__(self, rating_threshold=rating_threshold, exclude_unknowns=exclude_unknowns, verbose=verbose)

        self._data = data
        self._shuffle = shuffle
        self._train_size, self._val_size, self._test_size = self._validate_sizes(val_size, test_size, len(data))
        self._split = False


    def _validate_sizes(self, val_size, test_size, num_ratings):
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


    def _split_data(self):
        if self.verbose:
            print("Splitting the data")

        if self._shuffle and self._data is not None:
            shuffle(self._data)

        global_uid_map = {}
        global_iid_map = {}
        global_ur_set = set() # avoid duplicate rating in the data

        train_data = self._data[:self._train_size]
        self.train_set = MatrixTrainSet.from_triplets(train_data, global_uid_map, global_iid_map, global_ur_set, self.verbose)

        val_data = self._data[self._train_size:(self._train_size + self._val_size)]
        self.val_set = TestSet.from_triplets(val_data, global_uid_map, global_iid_map, global_ur_set, self.verbose)

        test_data = self._data[-self._test_size:]
        self.test_set = TestSet.from_triplets(test_data, global_uid_map, global_iid_map, global_ur_set, self.verbose)

        self._split = True
        self.total_users = len(global_uid_map)
        self.total_items = len(global_iid_map)

        if self.verbose:
            print('Total users = {}'.format(self.total_users))
            print('Total items = {}'.format(self.total_items))

        # free memory after splitting
        del self._data, global_uid_map, global_iid_map, global_ur_set


    def evaluate(self, model, metrics, user_based):
        if not self._split:
            self._split_data()

        return BaseStrategy.evaluate(self, model, metrics, user_based)
