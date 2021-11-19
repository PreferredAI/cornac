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

from cornac.eval_methods import PredefinedSplit
from cornac.data import Reader
from cornac.models import MF
from cornac.metrics import MAE, Recall


class TestPredefinedSplit(unittest.TestCase):

    def setUp(self):
        self.data = Reader().read('./tests/data.txt')


    def test_builds(self):
        try:
            PredefinedSplit(self.data, self.data[:8], self.data[8:], seed=123, verbose=True)
        except ValueError: # validation data is empty because unknowns are filtered
            assert True

        data = [(u, i, random.randint(1, 5))
                for (u, i) in itertools.product(['u1', 'u2', 'u3', 'u4'],
                                                ['i1', 'i2', 'i3', 'i4', 'i5'])]
        predef_split = PredefinedSplit(data, data[0:16], data[16:18], val_data = data[18:20], seed=123, verbose=True)

        self.assertTrue(predef_split.train_size == 16)
        self.assertTrue(predef_split.test_size == 2)
        self.assertTrue(predef_split.val_size == 2)

    def test_evaluate(self):
        ratio_split = PredefinedSplit(self.data, self.data[:8], self.data[8:], exclude_unknowns=False, verbose=True)
        ratio_split.evaluate(MF(), [MAE(), Recall()], user_based=False)

        ratio_split = PredefinedSplit(self.data, self.data[:8], self.data[8:], exclude_unknowns=False, verbose=True)
        ratio_split.evaluate(MF(), [MAE(), Recall()], user_based=False)

        users = []
        items = []
        for u, i, r in self.data:
            users.append(u)
            items.append(i)
        for u in users:
            for i in items:
                self.data.append((u, i, 5))

        ratio_split = PredefinedSplit(self.data, self.data[:8], self.data[8:], exclude_unknowns=False, verbose=True)
        ratio_split.evaluate(MF(), [MAE(), Recall()], user_based=True)


if __name__ == '__main__':
    unittest.main()
