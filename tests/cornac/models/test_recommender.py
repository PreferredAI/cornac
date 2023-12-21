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

from cornac.data import BasketDataset, Dataset, SequentialDataset, Reader
from cornac.models import MF, GPTop, SPop, NextBasketRecommender, NextItemRecommender


class TestRecommender(unittest.TestCase):
    def setUp(self):
        self.data = Reader().read("./tests/data.txt")

    def test_knows_x(self):
        mf = MF(1, 1, seed=123)
        dataset = Dataset.from_uir(self.data)
        mf.fit(dataset)

        self.assertTrue(mf.knows_user(7))
        self.assertFalse(mf.knows_item(13))

        self.assertTrue(mf.knows_item(3))
        self.assertFalse(mf.knows_item(16))

    def test_recommend(self):
        mf = MF(1, 1, seed=123)
        dataset = Dataset.from_uir(self.data)
        mf.fit(dataset)
        self.assertFalse(
            all(
                [
                    a == b
                    for a, b in zip(
                        mf.recommend("76", k=3, remove_seen=False),
                        mf.recommend("76", k=3, remove_seen=True, train_set=dataset),
                    )
                ]
            )
        )


class TestNextBasketRecommender(unittest.TestCase):
    def setUp(self):
        self.data = Reader().read("./tests/basket.txt", fmt="UBITJson")

    def test_init(self):
        model = NextBasketRecommender("test")
        self.assertTrue(model.name == "test")

    def test_fit(self):
        dataset = BasketDataset.from_ubi(self.data)
        model = NextBasketRecommender("")
        model.fit(dataset)
        model = GPTop()
        model.fit(dataset)
        model.score(0, [[]])
        model.rank(0, history_baskets=[[]])


class TestNextItemRecommender(unittest.TestCase):
    def setUp(self):
        self.data = Reader().read("./tests/sequence.txt", fmt="USIT", sep=" ")

    def test_init(self):
        model = NextItemRecommender("test")
        self.assertTrue(model.name == "test")

    def test_fit(self):
        dataset = SequentialDataset.from_usit(self.data)
        model = NextItemRecommender("")
        model.fit(dataset)
        model = SPop()
        model.fit(dataset)
        model.score(0, [])
        model.rank(0, history_items=[])


if __name__ == "__main__":
    unittest.main()
