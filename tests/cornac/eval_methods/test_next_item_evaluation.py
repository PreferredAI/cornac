# Copyright 2023 The Cornac Authors. All Rights Reserved.
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

from cornac.eval_methods import NextItemEvaluation
from cornac.data import Reader
from cornac.models import SPop
from cornac.metrics import HitRatio, Recall


class TestNextItemEvaluation(unittest.TestCase):
    def setUp(self):
        self.data = Reader().read("./tests/sequence.txt", fmt="USIT", sep=" ")

    def test_from_splits(self):
        next_item_eval = NextItemEvaluation.from_splits(train_data=self.data[:50], test_data=self.data[50:], fmt="USIT")

        self.assertTrue(next_item_eval.train_set != None)
        self.assertTrue(next_item_eval.test_set != None)
        self.assertTrue(next_item_eval.val_set == None)
        self.assertTrue(next_item_eval.total_sessions == 16)

    def test_evaluate(self):
        next_item_eval = NextItemEvaluation.from_splits(train_data=self.data[:50], test_data=self.data[50:], fmt="USIT")
        next_item_eval.evaluate(
            SPop(), [HitRatio(k=2), Recall(k=2)], user_based=True
        )

        next_item_eval = NextItemEvaluation.from_splits(train_data=self.data[:50], test_data=self.data[50:], fmt="USIT")
        next_item_eval.evaluate(
            SPop(), [HitRatio(k=2), Recall(k=2)], user_based=True
        )


if __name__ == "__main__":
    unittest.main()
