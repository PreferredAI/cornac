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

import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from cornac import Experiment
from cornac.data import BundleDataset, Dataset
from cornac.eval_methods import BundleConstructionEvaluation, BundleEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import (
    BPR,
    BundleConstructionRecommender,
    BundleRecommender,
    ItemKNN,
    MostPop,
)


CATALOG = [("b1", "b1"), ("b1", "i2"), ("b2", "i2"), ("b3", "i3")]
TRAIN = [("u1", "b1", 1), ("u2", "b2", 1)]
TEST = [("u1", "b2", 1), ("u2", "b3", 1)]


UB = [("0", "0", 1), ("1", "1", 1)]
BI = [("0", "0"), ("1", "1"), ("2", "0")]
UI = [("0", "0", 1), ("1", "1", 1), ("2", "2", 1)]
IDS = dict(
    user_ids=["0", "1", "2", "3"],
    bundle_ids=["0", "1", "2"],
    constituent_item_ids=["0", "1", "2", "3"],
)


class OrderedScore(BundleRecommender):
    def __init__(self):
        super().__init__("OrderedScore")

    def score(self, user_idx, item_idx=None):
        scores = np.array([0.0, 3.0, 2.0, 1.0])
        return scores if item_idx is None else scores[item_idx]


# Full training bundles; p1/p2 only add popularity to a/d without co-occurrence.
BIT_TRAIN = [
    ("b1", "a", 1),
    ("b1", "b", 2),
    ("b1", "c", 3),
    ("b2", "c", 1),
    ("b2", "d", 2),
    ("b3", "d", 1),
    ("b3", "e", 2),
    ("p1", "a", 1),
    ("p2", "d", 1),
]
# input_order="first", input_ratio=0.5: t1 gives a,b -> c; t2 gives d -> a;
# t3 gives a,c -> b,f. Item f appears only as a target.
BIT_TEST = [
    ("t1", "b", 2),
    ("t1", "a", 1),
    ("t1", "c", 3),
    ("t2", "a", 2),
    ("t2", "d", 1),
    ("t3", "b", 3),
    ("t3", "f", 4),
    ("t3", "a", 1),
    ("t3", "c", 2),
]


class ProbeMostPop(MostPop):
    """MostPop recording the candidate items it is asked to rank."""

    def __init__(self):
        super().__init__()
        self.candidates = {}

    def rank(self, user_idx, item_indices=None, k=-1, **kwargs):
        self.candidates.setdefault(user_idx, []).append(
            set(np.asarray(item_indices).tolist())
        )
        return super().rank(user_idx, item_indices, k, **kwargs)


class ToyConstruction(BundleConstructionRecommender):
    def __init__(self):
        super().__init__("ToyConstruction")
        self.calls = []

    def score(self, user_idx, item_idx=None):
        scores = np.arange(self.num_items, dtype=float)
        return scores if item_idx is None else scores[item_idx]

    def construct(self, user_idx, input_items, size, **kwargs):
        self.calls.append((user_idx, list(input_items), size))
        return super().construct(user_idx, input_items, size, **kwargs)


class TestBundleEvaluation(unittest.TestCase):
    def test_aligned_relations_and_no_leakage(self):
        evaluation = BundleEvaluation.from_splits(
            iter(TRAIN),
            iter(TEST),
            val_data=iter([("u2", "b1", 1)]),
            bundle_item_data=iter(CATALOG),
            user_item_data=iter([("u1", "i4", 2), ("new_user", "ignored", 1)]),
        )
        self.assertEqual(evaluation.total_bundles, 3)
        for dataset in (evaluation.train_set, evaluation.test_set, evaluation.val_set):
            self.assertIsInstance(dataset, BundleDataset)
            self.assertIs(dataset.bid_map, evaluation.global_iid_map)
            self.assertIs(
                dataset.constituent_iid_map, evaluation.train_set.constituent_iid_map
            )
            self.assertEqual(dataset.bundle_item_matrix.shape, (3, 4))
            self.assertEqual(dataset.user_item_matrix.nnz, 1)
        self.assertEqual(evaluation.train_set.matrix[0, 1], 0)
        self.assertEqual(evaluation.train_set.user_item_matrix[0, 1], 0)
        self.assertNotIn("ignored", evaluation.train_set.constituent_iid_map)

    def test_unknown_users_and_catalog_only_bundle(self):
        evaluation = BundleEvaluation.from_splits(
            TRAIN,
            TEST + [("unknown", "b1", 1)],
            bundle_item_data=CATALOG,
            user_item_data=[("unknown", "ignored", 1)],
            exclude_unknowns=True,
        )
        self.assertEqual(evaluation.test_set.num_ratings, 2)
        self.assertEqual(evaluation.train_set.num_bundles, 3)
        self.assertNotIn("unknown", evaluation.global_uid_map)
        self.assertNotIn("ignored", evaluation.train_set.constituent_iid_map)

    def test_allow_unknown_users(self):
        evaluation = BundleEvaluation.from_splits(
            TRAIN,
            TEST + [("unknown", "b1", 1)],
            bundle_item_data=CATALOG,
            user_item_data=[("unknown", "ignored", 1)],
            exclude_unknowns=False,
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
                TRAIN,
                [("u1", "missing", 1)],
                bundle_item_data=CATALOG,
            )
        with self.assertRaisesRegex(ValueError, "train_data is required"):
            BundleEvaluation.from_splits([], TEST, bundle_item_data=CATALOG)

    def test_rebuild_clears_old_validation(self):
        evaluation = BundleEvaluation.from_splits(
            TRAIN,
            TEST,
            val_data=[("u2", "b1", 1)],
            bundle_item_data=CATALOG,
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
        result, _ = evaluation.evaluate(
            model, [Recall(k=2), NDCG(k=2)], user_based=True
        )
        self.assertEqual(model.score(0).shape, (3,))
        self.assertTrue(np.isfinite(model.score(0)).all())
        for name in ("Recall@2", "NDCG@2"):
            self.assertTrue(np.isfinite(result.metric_avg_results[name]))
            self.assertGreaterEqual(result.metric_avg_results[name], 0)
            self.assertLessEqual(result.metric_avg_results[name], 1)

    def test_evaluation_preserves_full_catalog(self):
        method = BundleEvaluation.from_splits(
            UB,
            [("0", "2", 1)],
            val_data=[("1", "2", 1)],
            bundle_item_data=BI,
            user_item_data=UI,
            **IDS,
        )
        for data in (method.train_set, method.test_set, method.val_set):
            self.assertEqual(data.user_item_matrix.nnz, 3)
            self.assertEqual(data.user_item_matrix.shape, (4, 4))
            self.assertEqual(data.bundle_item_matrix.shape, (3, 4))
            self.assertEqual(data.uid_map["3"], 3)

    def test_reference_train_only_test_mask(self):
        membership = [(str(index), str(index)) for index in range(4)]
        for mode, recall in (("train_and_validation", 1), ("train", 0)):
            method = BundleEvaluation.from_splits(
                [("u", "0", 1)],
                [("u", "2", 1)],
                val_data=[("u", "1", 1)],
                bundle_item_data=membership,
                test_mask=mode,
            )
            model = OrderedScore()
            result, validation = method.evaluate(model, [Recall(k=1), NDCG(k=1)], True)
            self.assertEqual(result.metric_avg_results["Recall@1"], recall)
            self.assertEqual(result.metric_avg_results["NDCG@1"], recall)
            self.assertEqual(validation.metric_avg_results["Recall@1"], 1)
            self.assertIs(model.val_set, method.val_set)

    def test_undeclared_user_rejected(self):
        with self.assertRaises(ValueError):
            BundleEvaluation.from_splits(
                UB, [("bad", "2", 1)], bundle_item_data=BI, user_item_data=UI, **IDS
            )


class TestBundleConstructionEvaluation(unittest.TestCase):
    def test_seeded_bundle_split(self):
        data = [("b%d" % b, "i%d" % (b + j)) for b in range(10) for j in range(4)]
        evaluation = BundleConstructionEvaluation(
            data, test_size=0.2, val_size=0.1, seed=3
        )
        same = BundleConstructionEvaluation(
            iter(data), test_size=0.2, val_size=0.1, seed=3
        )
        self.assertEqual(list(evaluation.global_uid_map), list(same.global_uid_map))
        train_set, test_set, val_set = (
            evaluation.train_set,
            evaluation.test_set,
            evaluation.val_set,
        )
        self.assertEqual((evaluation.total_users, evaluation.total_items), (10, 13))
        self.assertEqual(len(set(test_set.uir_tuple[0])), 2)
        self.assertEqual(len(set(val_set.uir_tuple[0])), 1)
        assert_array_equal(train_set.partial_bundles, [7, 8, 9])
        for idx in range(7):
            self.assertEqual(len(train_set.bundle_items(idx)), 4)
        for idx in train_set.partial_bundles:
            given = set(train_set.bundle_items(idx))
            targets = set(test_set.bundle_items(idx)) | set(val_set.bundle_items(idx))
            self.assertEqual((len(given), len(targets)), (2, 2))
            self.assertFalse(given & targets)
            self.assertEqual(
                given | targets,
                {
                    evaluation.global_iid_map[
                        "i%d" % (int(train_set.bundle_ids[idx][1:]) + j)
                    ]
                    for j in range(4)
                },
            )
        for dataset in (test_set, val_set):
            self.assertIs(dataset.uid_map, train_set.uid_map)
            self.assertIs(dataset.iid_map, train_set.iid_map)
            assert_array_equal(dataset.owner_indices, [-1] * 10)
            self.assertEqual(len(dataset.partial_bundles), 0)
        self.assertIsNone(train_set.user_bundle_matrix)
        test_result, val_result = evaluation.evaluate(
            MostPop(), [Recall(k=2)], user_based=True
        )
        for result in (test_result, val_result):
            self.assertIn("Set-F1", result.metric_avg_results)
            self.assertIn("Set-Jaccard", result.metric_avg_results)
        evaluation.set_metrics = False
        test_result, _ = evaluation.evaluate(MostPop(), [Recall(k=2)], user_based=True)
        self.assertNotIn("Set-F1", test_result.metric_avg_results)

    def test_no_leakage_and_side_relations(self):
        train = [
            ("u1", "b1", "a"),
            ("u1", "b1", "b"),
            ("u2", "b2", "b"),
            ("u2", "b2", "c"),
        ]
        test = [
            ("u1", "t1", "a"),
            ("u1", "t1", "c"),
            ("u1", "t1", "d"),
            ("u1", "t1", "e"),
        ]
        user_bundle = [
            ("u1", "b1", 1),
            ("u3", "b1", 1),
            ("u2", "b2", 1),
            ("u3", "t1", 1),
            ("u1", "t1", 1),
        ]
        user_item = [("u4", "a", 2), ("u4", "a", 3), ("u1", "unknown", 1)]
        evaluation = BundleConstructionEvaluation.from_splits(
            train,
            test,
            fmt="UBI",
            user_bundle_data=user_bundle,
            user_item_data=user_item,
            seed=5,
        )
        train_set, test_set = evaluation.train_set, evaluation.test_set
        self.assertEqual(
            list(evaluation.global_uid_map), ["b1", "b2", ("u3", "t1"), ("u1", "t1")]
        )
        self.assertEqual(list(train_set.owner_map), ["u1", "u2", "u3", "u4"])
        assert_array_equal(train_set.owner_indices, [0, 1, 2, 0])
        assert_array_equal(train_set.partial_bundles, [2, 3])
        # Training links only: held-out t1 never reaches the user-bundle relation.
        assert_array_equal(
            train_set.user_bundle_matrix.toarray(),
            [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        )
        self.assertEqual(train_set.user_item_matrix.shape, (4, 5))
        self.assertEqual(train_set.user_item_matrix.nnz, 1)
        self.assertEqual(
            train_set.user_item_matrix[3, evaluation.global_iid_map["a"]], 2
        )
        self.assertIs(test_set.user_bundle_matrix, train_set.user_bundle_matrix)
        self.assertIs(test_set.user_item_matrix, train_set.user_item_matrix)
        given = train_set.bundle_items(2)
        assert_array_equal(train_set.bundle_items(3), given)
        assert_array_equal(test_set.bundle_items(3), test_set.bundle_items(2))
        targets = set(test_set.bundle_items(2))
        self.assertEqual(len(given), 2)
        self.assertEqual(set(given) | targets, {0, 2, 3, 4})
        train_pairs = set(zip(*train_set.uir_tuple[:2]))
        for idx in (2, 3):
            self.assertFalse({(idx, item) for item in targets} & train_pairs)
        self.assertNotIn("t1", evaluation.global_uid_map)

    def test_given_items_excluded_and_target_only_items_ranked(self):
        evaluation = BundleConstructionEvaluation.from_splits(
            BIT_TRAIN, BIT_TEST, fmt="BIT", input_order="first"
        )
        iid = evaluation.global_iid_map
        self.assertEqual(list(iid), ["a", "b", "c", "d", "e", "f"])
        self.assertNotIn(iid["f"], evaluation.train_set.uir_tuple[1])
        model = ProbeMostPop()
        evaluation.evaluate(model, [Recall(k=20)], user_based=True)
        expected = {
            "t1": {"c", "d", "e", "f"},
            "t2": {"a", "b", "c", "e", "f"},
            "t3": {"b", "d", "e", "f"},
        }
        for bid, items in expected.items():
            calls = model.candidates[evaluation.global_uid_map[bid]]
            self.assertEqual(len(calls), 2)  # ranking metrics and set metrics
            for candidates in calls:
                self.assertEqual(candidates, {iid[i] for i in items})

    def test_hand_checked_metrics(self):
        evaluation = BundleConstructionEvaluation.from_splits(
            BIT_TRAIN, BIT_TEST, fmt="BIT", input_order="first"
        )
        t1, t2, t3 = (evaluation.global_uid_map[b] for b in ("t1", "t2", "t3"))
        # Training counts with given rows: a4 b2 c3 d4 e1 f0. MostPop top-|T|:
        # t1 {d} vs {c}; t2 {a} vs {a}; t3 {d, b} vs {b, f}.
        # ItemKNN scores items that co-occur with given items in training rows:
        # t1 {c} vs {c}; t2 {c, e} (top-1 misses a); t3 {b, d} vs {b, f}.
        # Recall@20 exceeds every candidate list, so f (target only) is always found.
        expected = {
            "MostPop": {
                "Recall@2": {t1: 1, t2: 1, t3: 0.5},
                "Set-F1": {t1: 0, t2: 1, t3: 0.5},
                "Set-Jaccard": {t1: 0, t2: 1, t3: 1 / 3},
            },
            "ItemKNN": {
                "Recall@2": {t1: 1, t2: 0, t3: 0.5},
                "Set-F1": {t1: 1, t2: 0, t3: 0.5},
                "Set-Jaccard": {t1: 1, t2: 0, t3: 1 / 3},
            },
        }
        with tempfile.TemporaryDirectory() as path:
            experiment = Experiment(
                evaluation,
                [MostPop(), ItemKNN(verbose=False, seed=1)],
                [Recall(k=2), Recall(k=20)],
                save_dir=path,
            )
            experiment.run()
        for result in experiment.result:
            self.assertAlmostEqual(result.metric_avg_results["Recall@20"], 1.0)
            for name, values in expected[result.model_name].items():
                self.assertEqual(set(result.metric_user_results[name]), set(values))
                for idx, value in values.items():
                    self.assertAlmostEqual(result.metric_user_results[name][idx], value)
                self.assertAlmostEqual(
                    result.metric_avg_results[name], sum(values.values()) / 3
                )

    def test_construction_recommender_interface(self):
        evaluation = BundleConstructionEvaluation.from_splits(
            BIT_TRAIN, BIT_TEST, fmt="BIT", input_order="first"
        )
        model = ToyConstruction()
        result, _ = evaluation.evaluate(model, [Recall(k=2)], user_based=True)
        # Scores follow item indices a..f = 0..5; t3 constructs {f, e} for {b, f}.
        self.assertEqual(model.calls, [(5, [0, 1], 1), (6, [3], 1), (7, [0, 2], 2)])
        self.assertAlmostEqual(result.metric_avg_results["Set-F1"], 0.5 / 3)
        self.assertAlmostEqual(result.metric_avg_results["Set-Jaccard"], 1 / 9)
        assert_array_equal(model.construct(5, [0, 1], 10), [5, 4, 3, 2])
        assert_array_equal(model.construct(5, [], 0), [])
        ordinary = Dataset.from_uir(TRAIN)
        with self.assertRaisesRegex(ValueError, "train_set must"):
            ToyConstruction().fit(ordinary)
        with self.assertRaisesRegex(ValueError, "val_set must"):
            ToyConstruction().fit(evaluation.train_set, ordinary)

    def test_input_options(self):
        train, test = [("b1", "a"), ("b1", "b")], [("t1", "a"), ("t1", "c")]
        for fmt in ("BI", "UBI"):
            with self.assertRaisesRegex(ValueError, "requires BIT or UBIT"):
                BundleConstructionEvaluation.from_splits(
                    train, test, fmt=fmt, input_order="first"
                )
        with self.assertRaises(ValueError):
            BundleConstructionEvaluation.from_splits(train, test, fmt="UIR")
        with self.assertRaisesRegex(ValueError, "input_ratio"):
            BundleConstructionEvaluation.from_splits(train, test, input_ratio=1.5)
        with self.assertRaisesRegex(ValueError, "min_bundle_size"):
            BundleConstructionEvaluation.from_splits(train, test, min_bundle_size=1)
        with self.assertRaisesRegex(ValueError, "undeclared item"):
            BundleConstructionEvaluation.from_splits(train, test, item_ids=["a", "b"])
        declared = BundleConstructionEvaluation.from_splits(
            train, test, item_ids=["z", "c", "b", "a"]
        )
        self.assertEqual(list(declared.global_iid_map), ["z", "c", "b", "a"])

    def test_item_order_is_split_independent(self):
        # Item indices break score ties, so they must not encode split membership.
        first = BundleConstructionEvaluation.from_splits(
            [("b1", "z"), ("b1", "y")], [("t1", "a"), ("t1", "b")], seed=1
        )
        swapped = BundleConstructionEvaluation.from_splits(
            [("t1", "a"), ("t1", "b")], [("b1", "z"), ("b1", "y")], seed=1
        )
        self.assertEqual(list(first.global_iid_map), ["a", "b", "y", "z"])
        self.assertEqual(list(swapped.global_iid_map), list(first.global_iid_map))

    def test_min_bundle_size(self):
        train = [("b1", "a"), ("b1", "b")]
        test = [("t1", "a"), ("t1", "b"), ("t1", "c"), ("t2", "c"), ("t2", "d")]
        evaluation = BundleConstructionEvaluation.from_splits(
            train, test, min_bundle_size=3, seed=1
        )
        self.assertEqual(list(evaluation.global_uid_map), ["b1", "t2", "t1"])
        assert_array_equal(evaluation.train_set.bundle_items(1), [2, 3])
        assert_array_equal(evaluation.train_set.partial_bundles, [2])
        self.assertEqual(set(evaluation.test_set.uir_tuple[0]), {2})

    def test_predefined_inputs(self):
        train = [("b1", "a", 0), ("b1", "b", 1), ("b2", "c", 0), ("b2", "d", 1)]
        evaluation = BundleConstructionEvaluation.from_splits(
            train,
            [("t1", "c", 1), ("t1", "b", 2)],
            [("v1", "e", 2)],
            fmt="BIT",
            test_input_data=[("t1", "a", 0)],
            val_input_data=[("v1", "d", 0), ("v1", "c", 1)],
            input_ratio=0.9,
        )
        train_set = evaluation.train_set
        self.assertEqual(list(evaluation.global_uid_map), ["b1", "b2", "t1", "v1"])
        self.assertEqual(list(evaluation.global_iid_map), ["a", "b", "c", "d", "e"])
        assert_array_equal(train_set.bundle_items(2), [0])
        assert_array_equal(evaluation.test_set.bundle_items(2), [2, 1])
        assert_array_equal(train_set.bundle_items(3), [3, 2])
        assert_array_equal(evaluation.val_set.bundle_items(3), [4])
        assert_array_equal(train_set.partial_bundles, [2, 3])
        assert_array_equal(train_set.timestamps, [0, 1, 0, 1, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "without held-out targets"):
            BundleConstructionEvaluation.from_splits(
                train, [("t1", "c", 1)], fmt="BIT", test_input_data=[("x", "a", 0)]
            )

    def test_duplicate_bundles_across_splits(self):
        train, test = [("b1", "a"), ("b1", "b")], [("t1", "a"), ("t1", "c")]
        for args in ((train, train), (train, test, test)):
            with self.assertRaisesRegex(ValueError, "more than one split"):
                BundleConstructionEvaluation.from_splits(*args)

    def test_construction_instances(self):
        train = [("b1", "a"), ("b1", "b"), ("b2", "b"), ("b2", "c")]
        test = [
            ("t1", "a"),
            ("t1", "c"),
            ("t2", "b"),
            ("t2", "d"),
            ("t3", "c"),
            ("t3", "d"),
        ]
        user_bundle = [
            ("u1", "b1"),
            ("u2", "b2"),
            ("u1", "t1"),
            ("u1", "t2"),
            ("u2", "t2"),
        ]
        with self.assertWarnsRegex(UserWarning, "1 held-out bundles without users"):
            evaluation = BundleConstructionEvaluation.from_splits(
                train, test, user_bundle_data=user_bundle, input_ratio=0
            )
        train_set, test_set = evaluation.train_set, evaluation.test_set
        self.assertEqual(
            list(evaluation.global_uid_map),
            ["b1", "b2", ("u1", "t1"), ("u1", "t2"), ("u2", "t2")],
        )
        assert_array_equal(train_set.owner_indices, [-1, -1, 0, 0, 1])
        assert_array_equal(train_set.partial_bundles, [2, 3, 4])
        self.assertEqual(set(train_set.uir_tuple[0]), {0, 1})
        assert_array_equal(
            train_set.user_bundle_matrix.toarray(), [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
        )
        iid = evaluation.global_iid_map
        assert_array_equal(test_set.bundle_items(2), [iid["a"], iid["c"]])
        for idx in (3, 4):
            assert_array_equal(test_set.bundle_items(idx), [iid["b"], iid["d"]])
        result, _ = evaluation.evaluate(MostPop(), [Recall(k=2)], user_based=True)
        for name in ("Recall@2", "Set-F1", "Set-Jaccard"):
            self.assertEqual(set(result.metric_user_results[name]), {2, 3, 4})


if __name__ == "__main__":
    unittest.main()
