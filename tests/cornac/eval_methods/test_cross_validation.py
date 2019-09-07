# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import unittest

import numpy as np

from cornac.eval_methods import CrossValidation
from cornac.data import Reader


class TestCrossValidation(unittest.TestCase):

    def setUp(self):
        self.data = Reader().read('./tests/data.txt')
        self.n_folds = 5
        self.cv = CrossValidation(data=self.data, n_folds=self.n_folds, exclude_unknowns=False)

    def test_partition_data(self):
        ref_set = set(range(self.n_folds))
        res_set = set(self.cv._partition)
        fold_sizes = np.unique(self.cv._partition, return_counts=True)[1]

        self.assertEqual(len(self.data), len(self.cv._partition))
        self.assertEqual(res_set, ref_set)
        np.testing.assert_array_equal(fold_sizes, 2)

    def test_validate_partition(self):
        try:
            self.cv._validate_partition([0, 0, 1, 1])
        except ValueError:
            assert True

        try:
            self.cv._validate_partition([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
        except ValueError:
            assert True

    def test_get_train_test_sets_next_fold(self):
        for n in range(self.cv.n_folds):
            self.cv._get_train_test()
            self.assertEqual(self.cv.current_fold, n)
            self.assertSequenceEqual(self.cv.train_set.matrix.shape, (8, 8))
            self.cv._next_fold()


if __name__ == '__main__':
    unittest.main()
