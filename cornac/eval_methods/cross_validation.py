# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np
from .base_method import BaseMethod
from ..utils.generic_utils import safe_indexing
from ..experiment.cv_result import CVSingleModelResult


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

    exclude_unknowns: bool, optional, default: False
        Ignore unknown users and items (cold-start) during evaluation and testing

    verbose: bool, optional, default: False
        Output running log
    """

    def __init__(self, data, data_format='UIR', n_folds=5, rating_threshold=1., partition=None,
                 exclude_unknowns=True, verbose=False):
        BaseMethod.__init__(self, data=data, data_format=data_format, rating_threshold=rating_threshold,
                            exclude_unknowns=exclude_unknowns, verbose=verbose)
        self.n_folds = n_folds
        self.current_fold = 0
        self.current_split = None
        self.n_ratings = len(self._data)

        if partition is None:
            self.partition_data()
        else:
            self.partition = self._validate_partition(partition)

    # Partition ratings into n_folds
    def partition_data(self):

        fold_size = int(self.n_ratings / self.n_folds)
        remain_size = self.n_ratings - fold_size * self.n_folds

        self.partition = np.repeat(np.arange(self.n_folds), fold_size)

        if remain_size > 0:
            remain_partition = np.random.choice(self.n_folds, size=remain_size, replace=True, p=None)
            self.partition = np.concatenate((self.partition, remain_partition))

        np.random.shuffle(self.partition)

    def _validate_partition(self, partition):
        if len(partition) != self.n_ratings:
            raise Exception('The partition length must be equal to the number of ratings')
        elif len(set(partition)) != self.n_folds:
            raise Exception('Number of folds in given partition different from %s' % (self.n_folds))

        return partition

    def _get_train_test_sets(self):
        if self.verbose:
            print('Fold: {}'.format(self.current_fold + 1))

        test_idx = np.where(self.partition == self.current_fold)[0]
        train_idx = np.where(self.partition != self.current_fold)[0]

        train_data = safe_indexing(self._data, train_idx)
        test_data = safe_indexing(self._data, test_idx)

        if self.data_format == 'UIR':
            self._build_from_uir_format(train_data=train_data, test_data=test_data)

        if self.verbose:
            print('Total users = {}'.format(self.total_users))
            print('Total items = {}'.format(self.total_items))
            
    def _next_fold(self):
        if self.current_fold < self.n_folds - 1:
            self.current_fold = self.current_fold + 1
        else:
            self.current_fold = 0        

    def evaluate(self, model, metrics, user_based):
        result = CVSingleModelResult()

        for fold in range(self.n_folds):
            self._get_train_test_sets()
            avg_res, per_user_res = BaseMethod.evaluate(self, model, metrics, user_based)
            result._add_fold_res(fold=fold, metric_avg_results=avg_res)
            self._next_fold()
        result._compute_avg_res()
        return result
