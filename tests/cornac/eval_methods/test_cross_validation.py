# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import unittest
from cornac.eval_methods import CrossValidation
from cornac.data import Reader
import numpy as np


class TestCrossValidation(unittest.TestCase):

    def setUp(self):
        self.data = Reader().read('./tests/data.txt')
        self.n_folds = 5
        self.cv = CrossValidation(data=self.data, n_folds=self.n_folds)

    def test_partition_data(self):
        ref_set = set(range(self.n_folds))
        res_set = set(self.cv.partition)
        fold_sizes = np.unique(self.cv.partition, return_counts=True)[1]

        self.assertEqual(len(self.data), len(self.cv.partition))
        self.assertEqual(res_set, ref_set)
        np.testing.assert_array_equal(fold_sizes, 2)

    def test_validate_partition(self):
        try:
            self.cv._validate_partition([0, 0, 1, 1])
        except:
            assert True

        try:
            self.cv._validate_partition([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
        except:
            assert True

    def test_get_train_test_sets_next_fold(self):
        for n in range(self.cv.n_folds):
            self.cv._get_train_test()
            self.assertEqual(self.cv.current_fold, n)
            self.assertSequenceEqual(self.cv.train_set.matrix.shape, (8, 8))
            self.cv._next_fold()


if __name__ == '__main__':
    unittest.main()
