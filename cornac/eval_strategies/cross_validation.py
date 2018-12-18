# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np
from ..utils.util_functions import which_
from .eval_strategy import BaseStrategy
from .split import Split
from ..data import MatrixTrainSet, TestSet


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

    shuffle: bool, optional, default: True
        Shuffle the data before splitting.

    exclude_unknowns: bool, optional, default: False
        Ignore unknown users and items (cold-start) during evaluation and testing

    verbose: bool, optional, default: False
        Output running log
    """

    def __init__(self, data, n_folds=5, rating_threshold=1., partition=None, shuffle=True,
                 exclude_unknowns=False, verbose=False):
        BaseStrategy.__init__(self, rating_threshold=rating_threshold, exclude_unknowns=exclude_unknowns,
                                    verbose=verbose)
        self.n_folds = n_folds
        self.partition = partition
        self.current_fold = 0
        self.current_split = None
        self.n_ratings = data.shape[0]
		

    # Partition users into n_folds
    def _get_partition(self):

        n_fold_partition = np.random.choice(self.n_folds, size=self.n_ratings, replace=True,
                                            p=None)  # sample with replacement
        while len(set(n_fold_partition)) != self.n_folds:  # just in case some fold is empty
            n_fold_partition = np.random.choice(self.n_folds, size=self.n_ratings, replace=True, p=None)

        return n_fold_partition

    # This function is used to get the next train_test data
    def _get_next_train_test_split(self):
        #print(len(self.partition))
        index_test = np.where(self.partition == self.current_fold)[0]
        #print(len(index_test))
        index_train = np.where(self.partition != self.current_fold)[0]
        #print(len(index_train))
		
        global_uid_map = {}
        global_iid_map = {}
        global_ur_set = set() # avoid duplicate rating in the data

        train_data = self._data[index_train]
        self.train_set = MatrixTrainSet.from_triplets(train_data, global_uid_map, global_iid_map, global_ur_set, self.verbose)

        #val_data = self._data[self._train_size:(self._train_size + self._val_size)]
        #self.val_set = TestSet.from_triplets(val_data, global_uid_map, global_iid_map, global_ur_set, self.verbose)

        test_data = self._data[index_test]
        self.test_set = TestSet.from_triplets(test_data, global_uid_map, global_iid_map, global_ur_set, self.verbose)

        #self._split = True
        self.total_users = len(global_uid_map)
        self.total_items = len(global_iid_map)
		
		
        #self.current_split = Split(self.data, rating_threshold=self.rating_threshold, index_train=index_train,
        #                           index_test=index_test)

        if self.current_fold < self.n_folds - 1:
            self.current_fold = self.current_fold + 1
        else:
            self.current_fold = 0

    def evaluate(self, model, metrics):

        if self.partition is None:
            self.partition = self._get_partition()

        for fold in range(self.n_folds):
            print("fold:", self.current_fold)
            self._get_next_train_test_split()
            if self.current_fold == 1:
                res_tot = self.current_split.evaluate(model=model, metrics=metrics)
                resAvg = res_tot["ResAvg"]
                print(resAvg)
                resPerU = res_tot["ResPerUser"]
            else:
                res_tot = self.current_split.evaluate(model=model, metrics=metrics)
                """ need to figure out how to average the resuls accoording"""
                resAvg = np.vstack((resAvg, res_tot["ResAvg"]))
                resPerU = resPerU + res_tot["ResPerUser"]

        avg_resAvg = resAvg.mean(
            0)  # we are averaging the average results across the n_folds, another possibility is to make it per-user?
        std_resAvg = resAvg.std(0, ddof=1)

        # Averaging the results per-user across diffirent folds
        n_processed_u = resPerU[which_(resPerU[:, len(metrics)].todense().A1, ">", 0), len(metrics)].shape[0]
        resPerU[which_(resPerU[:, len(metrics)].todense().A1, ">", 0), :] = resPerU[which_(
            resPerU[:, len(metrics)].todense().A1, ">", 0), :] / resPerU[which_(resPerU[:, len(metrics)].todense().A1,
                                                                                ">", 0), len(
            metrics)].todense().reshape(n_processed_u, 1)

        # This is a temporary solution, we just return a single structure containing all the results, (may be consider returning an object of class result instead)
        res_tot = {"ResAvg": avg_resAvg[0:len(metrics)], "ResStd": std_resAvg[0:len(metrics)], "ResPerUser": resPerU}
        return res_tot
