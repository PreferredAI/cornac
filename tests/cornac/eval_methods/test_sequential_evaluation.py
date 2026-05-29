# Copyright 2026 The Cornac Authors. All Rights Reserved.
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

from cornac.data import Reader
from cornac.eval_methods import SequentialEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import SPop


class TestSequentialEvaluation(unittest.TestCase):
    def setUp(self):
        self.data = Reader().read("./tests/sequence.txt", fmt="USIT", sep=" ")

    def test_from_splits(self):
        eval_method = SequentialEvaluation.from_splits(
            train_data=self.data[:50], test_data=self.data[50:], fmt="USIT"
        )
        self.assertIsNotNone(eval_method.train_set)
        self.assertIsNotNone(eval_method.test_set)
        self.assertIsNone(eval_method.val_set)
        self.assertEqual(
            eval_method.total_sessions,
            eval_method.train_set.num_sessions + eval_method.test_set.num_sessions,
        )

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            SequentialEvaluation.from_splits(
                train_data=self.data[:50],
                test_data=self.data[50:],
                fmt="USIT",
                mode="bogus",
            )

    def _run_with_mode(self, mode):
        eval_method = SequentialEvaluation.from_splits(
            train_data=self.data[:50],
            test_data=self.data[50:],
            fmt="USIT",
            mode=mode,
        )
        test_result, _ = eval_method.evaluate(
            SPop(),
            [Recall(k=5), NDCG(k=5), MRR()],
            user_based=True,
        )
        return test_result.metric_avg_results

    def test_evaluate_mode_last(self):
        results = self._run_with_mode("last")
        for name, value in results.items():
            if name in ("Train (s)", "Test (s)"):
                continue
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_evaluate_mode_first(self):
        results = self._run_with_mode("first")
        for name, value in results.items():
            if name in ("Train (s)", "Test (s)"):
                continue
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_evaluate_mode_any(self):
        results = self._run_with_mode("any")
        for name, value in results.items():
            if name in ("Train (s)", "Test (s)"):
                continue
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
