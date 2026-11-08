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
import random
import time

from cornac.datasets import bundle


class TestBundle(unittest.TestCase):
    def test_load_youshu(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            self.assertEqual(len(bundle.load_feedback("Youshu", "train")), 34416)
            self.assertEqual(len(bundle.load_feedback("Youshu", "tune")), 5189)
            self.assertEqual(len(bundle.load_feedback("Youshu", "test")), 9746)
            self.assertEqual(len(bundle.load_user_item("Youshu")), 138515)
            self.assertEqual(len(bundle.load_bundle_item("Youshu")), 176667)
            ids = bundle.load_ids("Youshu")
            self.assertEqual(len(ids["user_ids"]), 8039)
            self.assertEqual(len(ids["bundle_ids"]), 4771)
            self.assertEqual(len(ids["constituent_item_ids"]), 32770)

    def test_load_spotify(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            self.assertEqual(len(bundle.load_feedback("Spotify", "train")), 72978)
            self.assertEqual(len(bundle.load_bundle_item("Spotify")), 555851)
            ids = bundle.load_ids("Spotify")
            self.assertEqual(len(ids["user_ids"]), 10183)
            self.assertEqual(len(ids["bundle_ids"]), 7787)
            self.assertEqual(len(ids["constituent_item_ids"]), 49434)
        with self.assertRaises(ValueError):
            bundle.load_user_item("Spotify")


if __name__ == "__main__":
    unittest.main()
