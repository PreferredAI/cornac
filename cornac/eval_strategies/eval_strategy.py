# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np


class EvaluationStrategy:
    """Evaluation Strategy (Generic Class)

    Parameters
    ----------
    data: scipy sparse matrix, required
        The user-item preference matrix.

    good_rating: float, optional, default: 1
        The minimum value that is considered to be a good rating, \
        e.g, if the ratings are in {1, ..., 5}, then good_rating = 4.

    data_train: ..., optional, default: None
        The training data.

    data_validation: ..., optional, default: None
        The validation data.

    data_test: ..., optional, default: None
        The test data.

    data_train_bin: ..., default: None
        The binary training data.

    data_validation_bin: ..., default: None
        The binary validation data.

    data_test_bin: ..., default: None
        The binary test data.

    data_nrows: int,
        The number of objects (users).

    data_ncols: int,
        The number of features (items).

    data_nnz: int,
        The number of observed ratings or the number of non-zero entries in data
    """

    def __init__(self, data, good_rating=1., data_train=None, data_validation=None, data_test=None):
        self._data = data
        self.good_rating = good_rating
        self._data_train = data_train
        self._data_validation = data_validation
        self._data_test = data_test

        self._data_train_bin = None
        self._data_validation_bin = None
        self._data_test_bin = None

        self.check_data_indices()

        # Useful attributes
        self.data_nrows = int(np.max(self.data[:, 0])) + 1
        self.data_ncols = int(np.max(self.data[:, 1])) + 1
        self.data_nnz = self.data.shape[0]  # the number of non-zero ratings

    @property
    def data(self):
        return self._data

    @property
    def data_train(self):
        return self._data_train

    @property
    def data_validation(self):
        return self._data_validation

    @property
    def data_test(self):
        return self._data_test

    @property
    def data_train_bin(self):
        return self._data_train_bin

    @property
    def data_validation_bin(self):
        return self._data_validation_bin

    @property
    def data_test_bin(self):
        return self._data_test_bin

        # check whether user and item indices start from 0 and update if necessary

    def check_data_indices(self):
        if np.min(self.data[:, 0]) > 0:
            self._data[:, 0] = self._data[:, 0] - np.min(self.data[:, 0])

        if np.min(self.data[:, 1]) > 0:
            self._data[:, 1] = self._data[:, 1] - np.min(self.data[:, 1])
