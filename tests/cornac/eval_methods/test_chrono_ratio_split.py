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
import itertools
import random
from math import isclose

from cornac.eval_methods import ChronoRatioSplit
from cornac.data import Reader
from cornac.models import MF
from cornac.metrics import MAE, Recall


class TestRatioSplit(unittest.TestCase):

    def setUp(self):
        self.data = Reader().read('./tests/data.txt', fmt='UIRT', sep='\t')

    def test_data_fmt(self):
        try:
            ChronoRatioSplit(self.data, fmt='UIR', verbose=True)
        except ValueError:
            assert True

        try:
            data = Reader().read('./tests/data.txt', fmt='UIR', sep='\t')
            ChronoRatioSplit(data, verbose=True)
        except ValueError:
            assert True

    def test_validate_size(self):
        train_size, val_size, test_size = ChronoRatioSplit.validate_size(0.1, 0.2)
        self.assertEqual(train_size, 0.7)
        self.assertEqual(val_size, 0.1)
        self.assertEqual(test_size, 0.2)

        train_size, val_size, test_size = ChronoRatioSplit.validate_size(None, 0.2)
        self.assertEqual(train_size, 0.8)
        self.assertEqual(val_size, 0.0)
        self.assertEqual(test_size, 0.2)

        train_size, val_size, test_size = ChronoRatioSplit.validate_size(None, None)
        self.assertEqual(train_size, 1.0)
        self.assertEqual(val_size, 0.0)
        self.assertEqual(test_size, 0.0)

        try:
            ChronoRatioSplit.validate_size(-0.1, 0.2)
        except ValueError:
            assert True

        try:
            ChronoRatioSplit.validate_size(2, 0.2)
        except ValueError:
            assert True

        try:
            ChronoRatioSplit.validate_size(0.1, -0.2)
        except ValueError:
            assert True

        try:
            ChronoRatioSplit.validate_size(0.1, 2)
        except ValueError:
            assert True

        try:
            ChronoRatioSplit.validate_size(0.6, 0.6)
        except ValueError:
            assert True

    def test_splits(self):
        try:
            ChronoRatioSplit(self.data, fmt='UIRT', test_size=0.1, val_size=0.1, verbose=True)
        except ValueError: # test_data and val_data are empty
            assert True

        data = [(u, i, random.randint(1, 5), random.randint(0, 100))
                for (u, i) in itertools.product(['u1', 'u2', 'u3', 'u4'],
                                                ['i1', 'i2', 'i3', 'i4', 'i5'])]
        ratio_split = ChronoRatioSplit(data, test_size=0.2, val_size=0.2, verbose=True)
        self.assertTrue(isclose(0.6, ratio_split.train_size))
        self.assertTrue(isclose(0.2, ratio_split.test_size))
        self.assertTrue(isclose(0.2, ratio_split.val_size))


if __name__ == '__main__':
    unittest.main()
