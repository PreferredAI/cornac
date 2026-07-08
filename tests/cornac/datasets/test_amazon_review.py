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

import gzip
import json
import os
import random
import tempfile
import time
import unittest

from cornac.datasets import amazon_review


class TestAmazonReview(unittest.TestCase):

    def test_preprocess_orders_chronologically_per_user(self):
        # Two users with out-of-order review times; one row missing a field.
        raw = [
            {"reviewerID": "u1", "asin": "iB", "overall": 4.0, "unixReviewTime": 200},
            {"reviewerID": "u1", "asin": "iA", "overall": 5.0, "unixReviewTime": 100},
            {"reviewerID": "u2", "asin": "iC", "overall": 3.0, "unixReviewTime": 150},
            {"reviewerID": "u1", "asin": "iD", "overall": 2.0},  # no timestamp -> dropped
        ]
        with tempfile.TemporaryDirectory() as d:
            gz_path = os.path.join(d, "reviews.json.gz")
            csv_path = os.path.join(d, "reviews.csv")
            with gzip.open(gz_path, "wt", encoding="utf-8") as f:
                for r in raw:
                    f.write(json.dumps(r) + "\n")

            amazon_review._preprocess(gz_path, csv_path)

            with open(csv_path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]

        # Missing-timestamp row dropped; u1 sorted by time (iA before iB).
        self.assertEqual(
            lines,
            ["u1,iA,5.0,100", "u1,iB,4.0,200", "u2,iC,3.0,150"],
        )

    def test_load_feedback(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            data = amazon_review.load_feedback(category="beauty")
            self.assertGreater(len(data), 0)
            self.assertEqual(len(data[0]), 4)  # (user, item, rating, timestamp)


if __name__ == "__main__":
    unittest.main()
