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

from cornac.data import BundleDataset, Dataset
from cornac.eval_methods import BundleEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import BPR, BundleRecommender


CATALOG = [("b1", "b1"), ("b1", "i2"), ("b2", "i2"), ("b3", "i3")]
TRAIN = [("u1", "b1", 1), ("u2", "b2", 1)]
TEST = [("u1", "b2", 1), ("u2", "b3", 1)]


class ToyBundle(BundleRecommender):
    def __init__(self):
        super().__init__("ToyBundle")

    def score(self, user_idx, item_idx=None):
        scores = np.arange(self.num_bundles, dtype=float)
        return scores if item_idx is None else scores[item_idx]


class TestBundleDataset(unittest.TestCase):
    def test_catalog_and_separate_ids(self):
        data = BundleDataset.build(TRAIN, CATALOG + [CATALOG[0]])
        self.assertEqual(data.matrix.shape, (2, 3))
        self.assertEqual(data.bundle_item_matrix.shape, (3, 3))
        self.assertEqual(data.bundle_item_matrix.nnz, 4)
        self.assertIs(data.bid_map, data.iid_map)
        self.assertIsNot(data.bid_map, data.constituent_iid_map)
        self.assertEqual(data.bundle_ids, ["b1", "b2", "b3"])
        self.assertEqual(data.num_constituent_items, 3)
        self.assertEqual(data.num_bundles, data.num_items)
        self.assertNotIn("i2", data.bid_map)
        self.assertNotIn("b2", data.constituent_iid_map)
        self.assertIsNone(data.user_item_matrix)
        assert_array_equal(data.bundle_item_matrix.toarray(),
                           [[1, 1, 0], [0, 1, 0], [0, 0, 1]])

    def test_independent_side_feedback(self):
        side = [("u1", "i4", 2), ("u1", "i4", 3), ("other", "ignored", 1)]
        data = BundleDataset.build(iter(TRAIN), iter(CATALOG), iter(side))
        self.assertEqual(data.user_item_matrix.shape, (2, 4))
        self.assertEqual(data.bundle_item_matrix.shape, (3, 4))
        self.assertEqual(data.user_item_matrix[0, data.constituent_iid_map["i4"]], 2)
        self.assertNotIn("ignored", data.constituent_iid_map)
        self.assertEqual(data.user_item_matrix.nnz, 1)

    def test_empty_side(self):
        for side in ([], [("other", "ignored", 1)]):
            data = BundleDataset.build(TRAIN, CATALOG, side)
            self.assertEqual(data.user_item_matrix.shape, (2, 3))
            self.assertEqual(data.user_item_matrix.nnz, 0)

    def test_invalid_catalog(self):
        with self.assertRaisesRegex(ValueError, "nonempty catalog"):
            BundleDataset.build(TRAIN, [])
        with self.assertRaisesRegex(ValueError, "absent from the catalog"):
            BundleDataset.build(TRAIN + [("u1", "missing", 1)], CATALOG)

    def test_shared_constituent_mapping_is_fixed(self):
        data = BundleDataset.build(TRAIN, CATALOG)
        before = dict(data.constituent_iid_map)
        with self.assertRaisesRegex(ValueError, "shared mapping"):
            BundleDataset.from_dataset(
                data, CATALOG, [("u1", "new_item", 1)],
                constituent_iid_map=data.constituent_iid_map,
            )
        self.assertEqual(dict(data.constituent_iid_map), before)
        self.assertEqual(data.bundle_item_matrix.shape, (3, 3))

    def test_timestamp_and_duplicate_feedback(self):
        with self.assertWarns(UserWarning):
            data = BundleDataset.build(
                [("u1", "b1", 1, 10), ("u1", "b1", 2, 20)],
                CATALOG, fmt="UIRT",
            )
        assert_array_equal(data.timestamps, [10])
        self.assertEqual(data.num_ratings, 1)


class TestBundleEvaluation(unittest.TestCase):
    def test_aligned_relations_and_no_leakage(self):
        evaluation = BundleEvaluation.from_splits(
            iter(TRAIN), iter(TEST), val_data=iter([("u2", "b1", 1)]),
            bundle_item_data=iter(CATALOG),
            user_item_data=iter([("u1", "i4", 2), ("new_user", "ignored", 1)]),
        )
        self.assertEqual(evaluation.total_bundles, 3)
        for dataset in (evaluation.train_set, evaluation.test_set, evaluation.val_set):
            self.assertIsInstance(dataset, BundleDataset)
            self.assertIs(dataset.bid_map, evaluation.global_iid_map)
            self.assertIs(dataset.constituent_iid_map,
                          evaluation.train_set.constituent_iid_map)
            self.assertEqual(dataset.bundle_item_matrix.shape, (3, 4))
            self.assertEqual(dataset.user_item_matrix.nnz, 1)
        self.assertEqual(evaluation.train_set.matrix[0, 1], 0)
        self.assertEqual(evaluation.train_set.user_item_matrix[0, 1], 0)
        self.assertNotIn("ignored", evaluation.train_set.constituent_iid_map)

    def test_unknown_users_and_catalog_only_bundle(self):
        evaluation = BundleEvaluation.from_splits(
            TRAIN, TEST + [("unknown", "b1", 1)], bundle_item_data=CATALOG,
            user_item_data=[("unknown", "ignored", 1)], exclude_unknowns=True,
        )
        self.assertEqual(evaluation.test_set.num_ratings, 2)
        self.assertEqual(evaluation.train_set.num_bundles, 3)
        self.assertNotIn("unknown", evaluation.global_uid_map)
        self.assertNotIn("ignored", evaluation.train_set.constituent_iid_map)

    def test_allow_unknown_users(self):
        evaluation = BundleEvaluation.from_splits(
            TRAIN, TEST + [("unknown", "b1", 1)], bundle_item_data=CATALOG,
            user_item_data=[("unknown", "ignored", 1)], exclude_unknowns=False,
        )
        self.assertEqual(evaluation.train_set.num_users, 2)
        self.assertEqual(evaluation.test_set.num_users, 3)
        self.assertEqual(evaluation.test_set.user_item_matrix.shape, (3, 3))
        self.assertEqual(evaluation.test_set.user_item_matrix.nnz, 0)
        self.assertNotIn("ignored", evaluation.train_set.constituent_iid_map)

    def test_invalid_and_empty_input(self):
        with self.assertRaises(ValueError):
            BundleEvaluation.from_splits(TRAIN, TEST, bundle_item_data=[])
        with self.assertRaisesRegex(ValueError, "absent from the catalog"):
            BundleEvaluation.from_splits(
                TRAIN, [("u1", "missing", 1)], bundle_item_data=CATALOG,
            )
        with self.assertRaisesRegex(ValueError, "train_data is required"):
            BundleEvaluation.from_splits([], TEST, bundle_item_data=CATALOG)

    def test_rebuild_clears_old_validation(self):
        evaluation = BundleEvaluation.from_splits(
            TRAIN, TEST, val_data=[("u2", "b1", 1)], bundle_item_data=CATALOG,
        )
        evaluation.build(TRAIN, TEST)
        self.assertIsNone(evaluation.val_set)
        self.assertEqual(evaluation.total_bundles, 3)

    def test_bpr_cutoff_exceeds_catalog(self):
        evaluation = BundleEvaluation.from_splits(TRAIN, TEST, bundle_item_data=CATALOG)
        model = BPR(k=2, max_iter=1, seed=7)
        result, _ = evaluation.evaluate(model, [Recall(k=20)], user_based=True)
        self.assertEqual(result.metric_avg_results["Recall@20"], 1.0)

    def test_bpr_ranking_metrics(self):
        evaluation = BundleEvaluation.from_splits(TRAIN, TEST, bundle_item_data=CATALOG)
        model = BPR(k=2, max_iter=2, seed=7)
        result, _ = evaluation.evaluate(model, [Recall(k=2), NDCG(k=2)], user_based=True)
        self.assertEqual(model.score(0).shape, (3,))
        self.assertTrue(np.isfinite(model.score(0)).all())
        for name in ("Recall@2", "NDCG@2"):
            self.assertTrue(np.isfinite(result.metric_avg_results[name]))
            self.assertGreaterEqual(result.metric_avg_results[name], 0)
            self.assertLessEqual(result.metric_avg_results[name], 1)


class TestBundleRecommender(unittest.TestCase):
    def test_inherited_prediction_and_persistence(self):
        data = BundleDataset.build(TRAIN, CATALOG)
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
        self.assertEqual(model.recommend("u2", remove_seen=True, train_set=data), ["b3", "b1"])
        assert_array_equal(model.rank(0, item_indices=np.array([0, 2]))[0], [2, 0])
        with tempfile.TemporaryDirectory() as path:
            loaded = ToyBundle.load(model.save(path))
            self.assertEqual(loaded.bundle_ids, model.bundle_ids)
            self.assertEqual(loaded.constituent_iid_map, model.constituent_iid_map)
            assert_array_equal(loaded.score(0), model.score(0))

    def test_dataset_requirement(self):
        ordinary = Dataset.from_uir(TRAIN)
        with self.assertRaisesRegex(ValueError, "train_set must"):
            ToyBundle().fit(ordinary)
        with self.assertRaisesRegex(ValueError, "val_set must"):
            ToyBundle().fit(BundleDataset.build(TRAIN, CATALOG), ordinary)
