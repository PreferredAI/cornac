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

from cornac.eval_methods import RatioSplit
from cornac.data import Reader
from cornac.models import MF
from cornac.metrics import MAE, Recall


class TestRatioSplit(unittest.TestCase):

    def setUp(self):
        self.data = Reader().read('./tests/data.txt')

    def test_validate_size(self):
        train_size, val_size, test_size = RatioSplit.validate_size(0.1, 0.2, 10)
        self.assertEqual(train_size, 7)
        self.assertEqual(val_size, 1)
        self.assertEqual(test_size, 2)

        train_size, val_size, test_size = RatioSplit.validate_size(None, 0.5, 10)
        self.assertEqual(train_size, 5)
        self.assertEqual(val_size, 0)
        self.assertEqual(test_size, 5)

        train_size, val_size, test_size = RatioSplit.validate_size(None, None, 10)
        self.assertEqual(train_size, 10)
        self.assertEqual(val_size, 0)
        self.assertEqual(test_size, 0)

        train_size, val_size, test_size = RatioSplit.validate_size(2, 2, 10)
        self.assertEqual(train_size, 6)
        self.assertEqual(val_size, 2)
        self.assertEqual(test_size, 2)

        try:
            RatioSplit.validate_size(-1, 0.2, 10)
        except ValueError:
            assert True

        try:
            RatioSplit.validate_size(1, -0.2, 10)
        except ValueError:
            assert True

        try:
            RatioSplit.validate_size(11, 0.2, 10)
        except ValueError:
            assert True

        try:
            RatioSplit.validate_size(0, 11, 10)
        except ValueError:
            assert True

        try:
            RatioSplit.validate_size(3, 8, 10)
        except ValueError:
            assert True

    def test_splits(self):
        try:
            RatioSplit(self.data, test_size=0.1, val_size=0.1, seed=123, verbose=True)
        except ValueError: # validation data is empty because unknowns are filtered
            assert True

        data = [(u, i, random.randint(1, 5))
                for (u, i) in itertools.product(['u1', 'u2', 'u3', 'u4'],
                                                ['i1', 'i2', 'i3', 'i4', 'i5'])]
        ratio_split = RatioSplit(data, test_size=0.1, val_size=0.1, seed=123, verbose=True)

        self.assertTrue(ratio_split.train_size == 16)
        self.assertTrue(ratio_split.test_size == 2)
        self.assertTrue(ratio_split.val_size == 2)

    def test_evaluate(self):
        ratio_split = RatioSplit(self.data, exclude_unknowns=False, verbose=True)
        ratio_split.evaluate(MF(), [MAE(), Recall()], user_based=False)

        ratio_split = RatioSplit(self.data, exclude_unknowns=False, verbose=True)
        ratio_split.evaluate(MF(), [MAE(), Recall()], user_based=False)

        users = []
        items = []
        for u, i, r in self.data:
            users.append(u)
            items.append(i)
        for u in users:
            for i in items:
                self.data.append((u, i, 5))

        ratio_split = RatioSplit(self.data, exclude_unknowns=False, verbose=True)
        ratio_split.evaluate(MF(), [MAE(), Recall()], user_based=True)


if __name__ == '__main__':
    unittest.main()
