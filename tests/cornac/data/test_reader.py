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

from cornac.data import Reader
from cornac.data.reader import read_text


class TestReader(unittest.TestCase):
    def setUp(self):
        self.data_file = "./tests/data.txt"
        self.basket_file = "./tests/basket.txt"
        self.reader = Reader()

    def test_raise(self):
        try:
            self.reader.read(self.data_file, fmt="bla bla")
        except ValueError:
            assert True

    def test_read_ui(self):
        triplets = self.reader.read(self.data_file, fmt="UI")
        self.assertEqual(len(triplets), 30)
        self.assertEqual(triplets[0][1], "93")
        self.assertEqual(triplets[1][2], 1.0)

        triplets = self.reader.read(self.data_file, fmt="UI", id_inline=True)
        self.assertEqual(len(triplets), 40)

    def test_read_uir(self):
        triplet_data = self.reader.read(self.data_file)

        self.assertEqual(len(triplet_data), 10)
        self.assertEqual(triplet_data[4][2], 3)
        self.assertEqual(triplet_data[6][1], "478")
        self.assertEqual(triplet_data[8][0], "543")

    def test_read_uirt(self):
        data = self.reader.read(self.data_file, fmt="UIRT")

        self.assertEqual(len(data), 10)
        self.assertEqual(data[4][3], 891656347)
        self.assertEqual(data[4][2], 3)
        self.assertEqual(data[4][1], "705")
        self.assertEqual(data[4][0], "329")
        self.assertEqual(data[9][3], 879451804)

    def test_read_tup(self):
        tup_data = self.reader.read(self.data_file, fmt="UITup")
        self.assertEqual(len(tup_data), 10)
        self.assertEqual(tup_data[4][2], [("3",), ("891656347",)])
        self.assertEqual(tup_data[6][1], "478")
        self.assertEqual(tup_data[8][0], "543")

    def test_read_review(self):
        review_data = self.reader.read("./tests/review.txt", fmt="UIReview")
        self.assertEqual(len(review_data), 5)
        self.assertEqual(review_data[0][2], "Sample text 1")
        self.assertEqual(review_data[1][1], "257")
        self.assertEqual(review_data[4][0], "329")

    def test_filter(self):
        reader = Reader(bin_threshold=4.0)
        data = reader.read(self.data_file)
        self.assertEqual(len(data), 8)
        self.assertListEqual([x[2] for x in data], [1] * len(data))

        reader = Reader(min_user_freq=2)
        self.assertEqual(len(reader.read(self.data_file)), 0)

        reader = Reader(min_item_freq=2)
        self.assertEqual(len(reader.read(self.data_file)), 0)

        reader = Reader(user_set=["76"], item_set=["93"])
        self.assertEqual(len(reader.read(self.data_file)), 1)

        reader = Reader(user_set=["76", "768"])
        self.assertEqual(len(reader.read(self.data_file)), 2)

        reader = Reader(item_set=["93", "257", "795"])
        self.assertEqual(len(reader.read(self.data_file)), 3)

    def test_read_text(self):
        self.assertEqual(len(read_text(self.data_file, sep=None)), 10)
        self.assertEqual(read_text(self.data_file, sep="\t")[1][0], "76")

    def test_read_basket(self):
        self.assertEqual(
            len(self.reader.read(self.basket_file, sep="\t", fmt="UBI")), 50
        )
        self.assertEqual(
            len(self.reader.read(self.basket_file, sep="\t", fmt="UBIT")), 50
        )
        self.assertEqual(
            len(self.reader.read(self.basket_file, sep="\t", fmt="UBITJson")), 50
        )


if __name__ == "__main__":
    unittest.main()
