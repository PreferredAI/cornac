# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np
from .base_strategy import BaseStrategy 
from ..utils.generic_utils import safe_indexing


class CrossValidation(BaseStrategy):
    """Evaluation Strategy Cross Validation.

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
        BaseStrategy.__init__(self, data=data, data_format = data_format, rating_threshold=rating_threshold,
                              exclude_unknowns=exclude_unknowns, verbose=verbose)
        self.n_folds = n_folds
        self.partition = partition
        self.current_fold = 0
        self.current_split = None
        self.n_ratings = self._data.shape[0]
		

    # Partition ratings into n_folds
    def _get_partition(self):

        n_fold_partition = np.random.choice(self.n_folds, size=self.n_ratings, replace=True,
                                            p=None)
        
        while len(set(n_fold_partition)) != self.n_folds:  # just in case some fold is empty
            n_fold_partition = np.random.choice(self.n_folds, size=self.n_ratings, replace=True, p=None)

        return n_fold_partition



    def _get_next_train_test_sets(self):
        
        if self.verbose:
            print('Fold: {}'.format(self.current_fold+1))

        test_idx = np.where(self.partition == self.current_fold)[0]
        train_idx = np.where(self.partition != self.current_fold)[0]
		
        train_data = safe_indexing(self._data, train_idx)
        test_data = safe_indexing(self._data, test_idx)
        val_data = np.array([])

        if self._data_format == 'UIR':
            self.build_from_uir_format(train_data, val_data, test_data)
		
		
        #self.current_split = Split(self.data, rating_threshold=self.rating_threshold, index_train=index_train,
        #                           index_test=index_test)

        if self.current_fold < self.n_folds - 1:
            self.current_fold = self.current_fold + 1
        else:
            self.current_fold = 0
            
        if self.verbose:
            print('Total users = {}'.format(self.total_users))
            print('Total items = {}'.format(self.total_items))

    def evaluate(self, model, metrics,user_based):

        per_fold_avg_res = {}
        per_fold_user_res = {}
        if self.partition is None:
            self.partition = self._get_partition()
        elif len(self.partition) != self.n_ratings:
            raise Exception('The partition length must be equal to the number of ratings')

        for fold in range(self.n_folds):
            self._get_next_train_test_sets()
            avg_res, per_user_res = BaseStrategy.evaluate(self, model, metrics, user_based)
            fold_name = 'fold:'+str(self.current_fold)
            per_fold_avg_res[fold_name] = avg_res
            per_fold_user_res[fold_name] = per_user_res

        return per_fold_avg_res, per_fold_user_res
