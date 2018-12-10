# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np
from ..utils.util_functions import which_
from .evaluation_strategy import EvaluationStrategy
from .split import Split


class CrossValidation(EvaluationStrategy):
    """Evaluation Strategy Cross Validation.

    Parameters
    ----------
    data: scipy sparse matrix, required
        The user-item interaction matrix.

    n_folds: int, optional, default: 5
        The number of folds for cross validation.

    good_rating: float, optional, default: 1
        The minimum value that is considered to be a good rating, \
        e.g, if the ratings are in {1, ... ,5}, then good_rating = 4.

    partition: array-like, shape (n_observed_ratings,), optional, default: None
        The partition of ratings into n_folds (fold label of each rating) \
        If None, random partitioning is performed to assign each rating into a fold.
    """

    def __init__(self, data, n_folds=5, good_rating=1., partition=None, data_train=None, data_validation=None,
                 data_test=None):
        EvaluationStrategy.__init__(self, data, good_rating=good_rating, data_train=data_train,
                                    data_validation=data_validation, data_test=data_test)
        self.n_folds = n_folds
        self.partition = partition
        self.current_fold = 0
        self.current_split = None

    # Partition users into n_folds
    def _get_partition(self):

        n_fold_partition = np.random.choice(self.n_folds, size=self.data_nnz, replace=True,
                                            p=None)  # sample with replacement
        while len(set(n_fold_partition)) != self.n_folds:  # just in case some fold is empty
            n_fold_partition = np.random.choice(self.n_folds, size=self.data_nnz, replace=True, p=None)

        return n_fold_partition

    # This function is used to get the next train_test data
    def _get_next_train_test_split(self):
        print(len(self.partition))
        index_test = np.where(self.partition == self.current_fold)[0]
        print(len(index_test))
        index_train = np.where(self.partition != self.current_fold)[0]
        print(len(index_train))
        self.current_split = Split(self.data, good_rating=self.good_rating, index_train=index_train,
                                   index_test=index_test)

        if self.current_fold < self.n_folds - 1:
            self.current_fold = self.current_fold + 1
        else:
            self.current_fold = 0

    def run_exp(self, model, metrics):

        if self.partition is None:
            self.partition = self._get_partition()

        for fold in range(self.n_folds):
            print("fold:", self.current_fold)
            self._get_next_train_test_split()
            if self.current_fold == 1:
                res_tot = self.current_split.run_exp(model=model, metrics=metrics)
                resAvg = res_tot["ResAvg"]
                print(resAvg)
                resPerU = res_tot["ResPerUser"]
            else:
                res_tot = self.current_split.run_exp(model=model, metrics=metrics)
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
