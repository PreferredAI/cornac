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

import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from cornac.data import BasketDataset, BundleDataset, Dataset, SequentialDataset, Reader
from cornac.models import (
    MF,
    GPTop,
    SPop,
    BundleRecommender,
    NextBasketRecommender,
    NextItemRecommender,
)


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

    def test_rank_cutoff_exceeds_candidates(self):
        dataset = Dataset.from_uir(self.data)
        model = MF(1, 1, seed=123).fit(dataset)
        for candidates in (None, np.array([0, 2]), np.array([1]),
                           np.array([], dtype=int)):
            expected, expected_scores = model.rank(0, item_indices=candidates)
            for cutoff in (model.total_items + 20, len(expected)):
                ranked, scores = model.rank(0, item_indices=candidates, k=cutoff)
                np.testing.assert_array_equal(ranked, expected)
                np.testing.assert_array_equal(scores, expected_scores)


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
        result = model.rank(0, history_items=[])
        self.assertTrue((result[0][0:3] == [3, 2, 4]).all())
        self.assertTrue((np.sort(result[0][3:5]) == [0, 1]).all()) # identical scores, sorting may affect the ordering
        self.assertTrue((result[0][5:6] == [5]).all())
        self.assertTrue((np.sort(result[0][6:9]) == [6, 7, 8]).all()) # identical scores, sorting may affect the ordering


BUNDLE_CATALOG = [("b1", "b1"), ("b1", "i2"), ("b2", "i2"), ("b3", "i3")]
BUNDLE_TRAIN = [("u1", "b1", 1), ("u2", "b2", 1)]


class ToyBundle(BundleRecommender):
    def __init__(self):
        super().__init__("ToyBundle")

    def score(self, user_idx, item_idx=None):
        scores = np.arange(self.num_bundles, dtype=float)
        return scores if item_idx is None else scores[item_idx]


class TestBundleRecommender(unittest.TestCase):
    def test_inherited_prediction_and_persistence(self):
        data = BundleDataset.build(BUNDLE_TRAIN, BUNDLE_CATALOG)
        model = ToyBundle().fit(data)
        self.assertEqual(model.num_bundles, 3)
        self.assertEqual(model.total_bundles, 3)
        self.assertEqual(model.num_constituent_items, 3)
        self.assertIs(model.bid_map, data.bid_map)
        self.assertEqual(model.score(0, 2), 2)
        assert_array_equal(model.rank(0, k=20)[0], [2, 1, 0])
        candidates = np.array([0, 2])
        assert_array_equal(model.rank(0, item_indices=candidates, k=20)[0], [2, 0])
        assert_array_equal(model.rank(0, item_indices=candidates, k=-1)[0], [2, 0])
        self.assertEqual(model.recommend("u1", k=2), ["b3", "b2"])
        self.assertEqual(
            model.recommend("u2", remove_seen=True, train_set=data), ["b3", "b1"]
        )
        assert_array_equal(model.rank(0, item_indices=np.array([0, 2]))[0], [2, 0])
        with tempfile.TemporaryDirectory() as path:
            loaded = ToyBundle.load(model.save(path))
            self.assertEqual(loaded.bundle_ids, model.bundle_ids)
            self.assertEqual(loaded.constituent_iid_map, model.constituent_iid_map)
            assert_array_equal(loaded.score(0), model.score(0))

    def test_dataset_requirement(self):
        ordinary = Dataset.from_uir(BUNDLE_TRAIN)
        with self.assertRaisesRegex(ValueError, "train_set must"):
            ToyBundle().fit(ordinary)
        with self.assertRaisesRegex(ValueError, "val_set must"):
            ToyBundle().fit(BundleDataset.build(BUNDLE_TRAIN, BUNDLE_CATALOG), ordinary)


if __name__ == "__main__":
    unittest.main()
