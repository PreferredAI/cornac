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

    def test_item_text(self):
        meta = {
            "asin": "iA",
            "title": "  Long   Wig ",
            "price": 11.83,
            "brand": "Generic",
            "categories": [["Beauty", "Hair Care"], ["Beauty", "Wigs"]],
        }
        self.assertEqual(
            amazon_review._item_text(meta),
            "Title: Long Wig. Price: 11.83. Brand: Generic. "
            "Categories: Beauty, Hair Care, Wigs",
        )
        # missing fields are skipped, not rendered empty
        self.assertEqual(amazon_review._item_text({"title": "X"}), "Title: X")

    def test_item_text_with_description(self):
        meta = {
            "asin": "iA",
            "title": "Wig",
            "description": "  A   long  soft   wig. ",
        }
        # description is appended last and whitespace-normalized
        self.assertEqual(
            amazon_review._item_text(meta, include_description=True),
            "Title: Wig. Description: A long soft wig.",
        )
        # missing/empty description is skipped
        self.assertEqual(
            amazon_review._item_text({"title": "Wig"}, include_description=True),
            "Title: Wig",
        )
        self.assertEqual(
            amazon_review._item_text({"title": "Wig", "description": ""}, include_description=True),
            "Title: Wig",
        )
        # include_description=False is unchanged even when description present
        self.assertEqual(amazon_review._item_text(meta), "Title: Wig")

    def test_preprocess_meta_covers_all_review_items(self):
        # Metadata lines are Python dict literals (single quotes), only items
        # in the reviews file are kept, and items without metadata get "".
        with tempfile.TemporaryDirectory() as d:
            reviews_csv = os.path.join(d, "reviews.csv")
            with open(reviews_csv, "w") as f:
                f.write("u1,iA,5.0,100\nu1,iB,4.0,200\nu2,iA,3.0,150\n")

            meta_gz = os.path.join(d, "meta.json.gz")
            with gzip.open(meta_gz, "wt", encoding="utf-8") as f:
                f.write("{'asin': 'iA', 'title': 'Item A', 'price': 9.99}\n")
                f.write("{'asin': 'iZ', 'title': 'Not reviewed'}\n")

            out_path = os.path.join(d, "text.csv")
            amazon_review._preprocess_meta(meta_gz, reviews_csv, out_path)

            with open(out_path) as f:
                rows = [ln.rstrip("\n").split(",", 1) for ln in f]

        self.assertEqual(rows[0], ["iA", "Title: Item A. Price: 9.99"])
        self.assertEqual(rows[1], ["iB", ""])  # covered, but no metadata
        self.assertEqual(len(rows), 2)  # iZ excluded

    def test_preprocess_meta_include_description(self):
        # include_description plumbs through to the emitted text.
        with tempfile.TemporaryDirectory() as d:
            reviews_csv = os.path.join(d, "reviews.csv")
            with open(reviews_csv, "w") as f:
                f.write("u1,iA,5.0,100\n")

            meta_gz = os.path.join(d, "meta.json.gz")
            with gzip.open(meta_gz, "wt", encoding="utf-8") as f:
                f.write("{'asin': 'iA', 'title': 'Item A', 'description': 'Nice item'}\n")

            out_path = os.path.join(d, "text.csv")
            amazon_review._preprocess_meta(meta_gz, reviews_csv, out_path, include_description=True)

            with open(out_path) as f:
                rows = [ln.rstrip("\n").split(",", 1) for ln in f]

        self.assertEqual(rows[0], ["iA", "Title: Item A. Description: Nice item"])


if __name__ == "__main__":
    unittest.main()
