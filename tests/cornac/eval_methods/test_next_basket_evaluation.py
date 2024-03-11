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

from cornac.eval_methods import NextBasketEvaluation
from cornac.data import Reader
from cornac.models import GPTop
from cornac.metrics import HitRatio, Recall


class TestNextBasketEvaluation(unittest.TestCase):
    def setUp(self):
        self.data = Reader().read("./tests/basket.txt", fmt="UBITJson", sep="\t")

    def test_splits(self):
        next_basket_eval = NextBasketEvaluation(
            self.data, test_size=0.1, val_size=0.1, seed=123, verbose=True
        )

        self.assertTrue(next_basket_eval.train_size == 8)
        self.assertTrue(next_basket_eval.test_size == 1)
        self.assertTrue(next_basket_eval.val_size == 1)

    def test_evaluate(self):
        next_basket_eval = NextBasketEvaluation(
            self.data, exclude_unknowns=False, verbose=True
        )
        next_basket_eval.evaluate(
            GPTop(), [HitRatio(k=2), Recall(k=2)], user_based=True
        )

        next_basket_eval = NextBasketEvaluation(
            self.data,
            repetition_eval=True,
            exploration_eval=True,
            exclude_unknowns=False,
            verbose=True,
        )
        next_basket_eval.evaluate(
            GPTop(), [HitRatio(k=2), Recall(k=2)], user_based=True
        )


if __name__ == "__main__":
    unittest.main()
