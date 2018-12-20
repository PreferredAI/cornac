# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..utils.util_functions import safe_indexing
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

    def __init__(self, data, data_format='UIR', val_size=0.0, test_size=0.2, rating_threshold=1., shuffle=True, random_state=None,
                 exclude_unknowns=False, verbose=False):
        super().__init__(self, rating_threshold=rating_threshold, exclude_unknowns=exclude_unknowns, verbose=verbose)

        self._data = data
        self._data_format = self._validate_data_format(data_format)
        self._shuffle = shuffle
        self._random_state = random_state
        self._train_size, self._val_size, self._test_size = self._validate_sizes(val_size, test_size, len(data))
        self._split = False


    @staticmethod
    def _validate_data_format(data_format):
        data_format = str(data_format).upper()
        if not data_format in ['UIR', 'UIRT']:
            raise ValueError('{} data format is not supported!'.format(data_format))

        return data_format


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
    
    
    def build_train_test_val(self,train_idx, test_idx, val_idx, data_format = 'UIR'):
        
        train_data = safe_indexing(self._data, train_idx)
        val_data = safe_indexing(self._data, val_idx)
        test_data = safe_indexing(self._data, test_idx)

        if self._data_format == 'UIR':
            self.build_from_uir_format(train_data, val_data, test_data)


    def build_from_uir_format(self, train_data, val_data, test_data):
        global_uid_map = {}
        global_iid_map = {}
        global_ui_set = set() # avoid duplicate ratings in the data

        if self.verbose:
            print('Building training set')
        self.train_set = MatrixTrainSet.from_uir_triplets(train_data, global_uid_map, global_iid_map, global_ui_set, self.verbose)

        if self.verbose:
            print('Building validation set')
        self.val_set = TestSet.from_uir_triplets(val_data, global_uid_map, global_iid_map, global_ui_set, self.verbose)

        if self.verbose:
            print('Building test set')
        self.test_set = TestSet.from_uir_triplets(test_data, global_uid_map, global_iid_map, global_ui_set, self.verbose)

        self.total_users = len(global_uid_map)
        self.total_items = len(global_iid_map)


    def _split_data(self):
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

        self.build_train_test_val(self,train_idx, val_idx, test_idx)

        self._split = True

        if self.verbose:
            print('Total users = {}'.format(self.total_users))
            print('Total items = {}'.format(self.total_items))


    def evaluate(self, model, metrics, user_based):
        if not self._split:
            self._split_data()

        return BaseStrategy.evaluate(self, model, metrics, user_based)
