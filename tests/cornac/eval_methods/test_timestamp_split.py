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

from cornac.eval_methods import TimestampSplit


class TestTimestampSplit(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        users = ["u1", "u2", "u3", "u4", "u5", "u6"]
        items = ["i1", "i2", "i3", "i4", "i5", "i6"]
        pairs = list(itertools.product(users, items))
        random.shuffle(pairs)  # spread users/items across the timeline
        self.data = [
            (u, i, random.randint(1, 5), ts)
            for ts, (u, i) in enumerate(pairs)
        ]  # timestamps 0..35

    def test_split(self):
        eval_method = TimestampSplit(
            self.data,
            val_timestamp=12,
            test_timestamp=24,
            exclude_unknowns=False,
            verbose=True,
        )
        self.assertEqual(eval_method.val_timestamp, 12)
        self.assertEqual(eval_method.test_timestamp, 24)
        self.assertEqual(eval_method.train_set.num_ratings, 12)
        self.assertEqual(eval_method.val_set.num_ratings, 12)
        self.assertEqual(eval_method.test_set.num_ratings, 12)

    def test_empty_val(self):
        # cutoffs land between integer timestamps, so val window contains no rows
        eval_method = TimestampSplit(
            self.data,
            val_timestamp=11.5,
            test_timestamp=11.6,
            exclude_unknowns=False,
            verbose=True,
        )
        self.assertEqual(eval_method.train_set.num_ratings, 12)
        self.assertIsNone(eval_method.val_set)
        self.assertEqual(eval_method.test_set.num_ratings, 24)

    def test_requires_uirt(self):
        uir_data = [(u, i, r) for (u, i, r, _) in self.data]
        with self.assertRaises(ValueError):
            TimestampSplit(uir_data, val_timestamp=12, test_timestamp=24, fmt="UIR")

    def test_missing_cutoffs(self):
        with self.assertRaises(ValueError):
            TimestampSplit(self.data, val_timestamp=None, test_timestamp=24)
        with self.assertRaises(ValueError):
            TimestampSplit(self.data, val_timestamp=12, test_timestamp=None)

    def test_invalid_order(self):
        with self.assertRaises(ValueError):
            TimestampSplit(self.data, val_timestamp=24, test_timestamp=12)
        with self.assertRaises(ValueError):
            TimestampSplit(self.data, val_timestamp=12, test_timestamp=12)

    def test_empty_train(self):
        with self.assertRaises(ValueError):
            TimestampSplit(self.data, val_timestamp=0, test_timestamp=10)

    def test_empty_test(self):
        with self.assertRaises(ValueError):
            TimestampSplit(self.data, val_timestamp=10, test_timestamp=100)


if __name__ == "__main__":
    unittest.main()
