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
    """

    def __init__(self, triplet_data, val_size=0.0, test_size=0.2, rating_threshold=1., shuffle=True):
        BaseStrategy.__init__(self, rating_threshold=rating_threshold)
        self._triplet_data = triplet_data
        self._shuffle = shuffle
        self._train_size, self._val_size, self._test_size = self._validate_sizes(val_size, test_size, len(triplet_data))
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
        print("Splitting the data")

        if self._shuffle and self._triplet_data is not None:
            shuffle(self._triplet_data)

        if self.train_set is None:
            train_data = self._triplet_data[:self._train_size]
            self.train_set = MatrixTrainSet.from_triplets(train_data)

        if self.val_set is None:
            val_data = self._triplet_data[self._train_size:(self._train_size + self._val_size)]
            self.val_set = TestSet.from_triplets(val_data)

        if self.test_set is None:
            test_data = self._triplet_data[-self._test_size:]
            self.test_set = TestSet.from_triplets(test_data)

        self._split = True
        del self._triplet_data # free memory after splitting

    def evaluate(self, model, metrics):
        if not self._split:
            self._split_data()

        return BaseStrategy.evaluate(self, model, metrics)
