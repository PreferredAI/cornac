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

import numpy as np

from cornac.data import FeatureModality
from cornac.eval_methods import BundleConstructionEvaluation
from cornac.exception import ScoreException
from cornac.metrics import Recall
from cornac.models import DDBC

try:
    import torch
except ImportError:
    torch = None

N_ITEMS = 12
ITEM_IDS = [str(i) for i in range(N_ITEMS)]


def bundle_rows(first, last):
    return [
        (str(b), str((b + p) % N_ITEMS), p)
        for b in range(first, last)
        for p in range(4)
    ]


def make_eval_method(**kwargs):
    features = np.random.RandomState(0).randn(N_ITEMS, 8)
    params = dict(
        train_data=bundle_rows(0, 8),
        val_data=bundle_rows(8, 10),
        test_data=bundle_rows(10, 12),
        fmt="BIT",
        input_ratio=0.5,
        input_order="first",
        item_ids=ITEM_IDS,
        exclude_unknowns=False,
        seed=123,
        item_feature=FeatureModality(features=features, ids=ITEM_IDS),
    )
    params.update(kwargs)
    return BundleConstructionEvaluation.from_splits(**params)


def tiny_ddbc(**kwargs):
    params = dict(
        bundle_size=4,
        n_codebooks=3,
        codebook_size=4,
        kmeans_iters=2,
        rvq_steps=2,
        hidden_size=8,
        cond_dim=8,
        n_blocks=1,
        n_heads=2,
        dropout=0.0,
        n_diffusion_steps=4,
        batch_size=4,
        accumulate=1,
        max_steps=3,
        warmup_steps=1,
        val_every=2,
        seed=1,
    )
    params.update(kwargs)
    return DDBC(**params)


class TestDDBCArguments(unittest.TestCase):
    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            DDBC(bundle_size=1)
        with self.assertRaises(ValueError):
            DDBC(decode="beam")
        with self.assertRaises(ValueError):
            DDBC(model_selection="first")
        with self.assertRaises(ValueError):
            DDBC(hidden_size=10, n_heads=4)

    def test_default_decode_follows_algorithm_2(self):
        self.assertEqual(DDBC().decode, "confidence")

    def test_paper_configs(self):
        from cornac.models.ddbc import (
            DDBC_POG_CONFIG,
            DDBC_SPOTIFY30_CONFIG,
            DDBC_SPOTIFY60_CONFIG,
            DDBC_SPOTIFY90_CONFIG,
        )

        expected = [
            (DDBC_POG_CONFIG, 4, 128, 0.8),
            (DDBC_SPOTIFY30_CONFIG, 30, 256, 0.4),
            (DDBC_SPOTIFY60_CONFIG, 60, 256, 0.4),
            (DDBC_SPOTIFY90_CONFIG, 90, 256, 0.4),
        ]
        for config, bundle_size, codebook_size, swap_ratio in expected:
            model = DDBC(**config)
            self.assertEqual(
                (model.bundle_size, model.codebook_size, model.swap_ratio),
                (bundle_size, codebook_size, swap_ratio),
            )

    def test_not_fitted(self):
        with self.assertRaises(ValueError):
            DDBC().construct(0, [0], 1)
        with self.assertRaises(ValueError):
            DDBC().score(0)


@unittest.skipIf(torch is None, "torch is not installed")
class TestDDBC(unittest.TestCase):
    def setUp(self):
        self.eval_method = make_eval_method()
        self.train_set = self.eval_method.train_set
        self.query = int(self.eval_method.test_set.uir_tuple[0][0])
        self.given = self.train_set.bundle_items(self.query)

    def fitted(self, **kwargs):
        return tiny_ddbc(**kwargs).fit(self.train_set, self.eval_method.val_set)

    def test_semantic_id_tokens(self):
        from cornac.models.ddbc.ddbc import semantic_id_tokens

        tokens = semantic_id_tokens(np.array([[0, 0], [0, 0], [1, 1]]), codebook_size=2)
        np.testing.assert_array_equal(tokens, [[1, 3, 5], [1, 3, 6], [2, 4, 5]])
        with self.assertRaises(ValueError):
            semantic_id_tokens(np.zeros((3, 2), dtype="int"), codebook_size=2)

    def test_swap_windows(self):
        from cornac.models.ddbc.ddbc import swap_windows

        windows = swap_windows(
            [[0, 1, 2, 3, 4], [5, 6]], 0.8, 4, np.random.RandomState(0)
        )

        self.assertEqual(len(windows), 4)
        for w in windows:
            self.assertEqual(len(w), 4)
            self.assertTrue(set(w) <= {0, 1, 2, 3, 4})

    def test_training_uses_full_bundles_only(self):
        bundles = DDBC._training_bundles(self.train_set)

        self.assertEqual(len(bundles), 8)
        self.assertEqual({len(b) for b in bundles}, {4})
        self.assertEqual(len(set(self.train_set.partial_bundles.tolist())), 4)

    def test_validation_bundles_are_rebuilt_in_order(self):
        iid_map = self.train_set.iid_map
        expected = [
            [iid_map[str((b + p) % N_ITEMS)] for p in range(4)] for b in range(8, 10)
        ]

        self.assertEqual(
            DDBC._held_out_bundles(self.train_set, self.eval_method.val_set), expected
        )

    def test_construct(self):
        model = self.fitted()

        items = model.construct(self.query, self.given, 2)
        self.assertEqual(len(items), 2)
        self.assertEqual(len(set(items.tolist())), 2)
        self.assertFalse(set(items.tolist()) & set(self.given.tolist()))
        self.assertTrue(((items >= 0) & (items < N_ITEMS)).all())
        np.testing.assert_array_equal(items, model.construct(self.query, self.given, 2))
        self.assertIsNotNone(model.best_val_nll)

        self.assertEqual(len(model.construct(self.query, self.given, 0)), 0)
        self.assertEqual(
            len(model.construct(self.query, self.given, N_ITEMS)), N_ITEMS - 2
        )

        model.decode = "ancestral"
        self.assertEqual(len(model.construct(self.query, self.given, 2)), 2)

    def test_codes_to_items(self):
        model = self.fitted()
        tokens = model.item_tokens

        np.testing.assert_array_equal(
            model._codes_to_items(tokens[[3, 5]], [0, 1]), [3, 5]
        )

        replaced = model._codes_to_items(tokens[[3, 3]], [0])
        self.assertEqual(replaced[0], 3)
        self.assertNotIn(replaced[1], [0, 3])
        similarity = model.item_units @ model.item_units[3]
        similarity[[0, 3]] = -np.inf
        self.assertEqual(replaced[1], similarity.argmax())

        unknown = tokens[3].copy()
        unknown[-1] = tokens[:, -1].max() + 1
        self.assertNotIn(tuple(unknown.tolist()), model.code_to_item)
        vector = model.code_vectors[unknown[:-1] - 1].sum(0)
        similarity = model.item_units @ (vector / np.linalg.norm(vector))
        similarity[[0]] = -np.inf
        self.assertEqual(
            model._codes_to_items(unknown[None], [0])[0], similarity.argmax()
        )

    def test_score(self):
        model = self.fitted()

        scores = model.score(self.query)
        self.assertEqual(scores.shape, (N_ITEMS,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(model.score(self.query, 3), scores[3])

        with self.assertRaises(ScoreException):
            model.score(0)
        with self.assertRaises(ScoreException):
            model.score(self.query, N_ITEMS)

    def test_epoch_callbacks(self):
        epochs = []
        model = tiny_ddbc()
        model.callbacks = [lambda m, epoch, logs: epochs.append((epoch, logs))]
        model.fit(self.train_set, self.eval_method.val_set)

        self.assertEqual([e for e, _ in epochs], list(range(1, len(epochs) + 1)))
        self.assertTrue(all(np.isfinite(logs["loss"]) for _, logs in epochs))
        val_scores = [logs["val_score"] for _, logs in epochs if "val_score" in logs]
        self.assertGreater(len(val_scores), 0)
        self.assertEqual(max(val_scores), -model.best_val_nll)

    def test_item_features_follow_the_catalog(self):
        model = self.fitted()
        features = np.random.RandomState(0).randn(N_ITEMS, 8)
        expected = features / np.linalg.norm(features, axis=1, keepdims=True)

        order = [self.train_set.iid_map[i] for i in ITEM_IDS]
        np.testing.assert_allclose(model.item_units[order], expected, atol=1e-5)
        self.assertEqual(len(model.item_tokens), N_ITEMS)

    def test_construction_from_scratch(self):
        eval_method = make_eval_method(
            input_ratio=0,
            input_order="random",
            fmt="UBIT",
            train_data=[("u", *row) for row in bundle_rows(0, 8)],
            val_data=None,
            test_data=[("u", *row) for row in bundle_rows(10, 12)],
        )
        query = int(eval_method.test_set.uir_tuple[0][0])
        self.assertEqual(len(eval_method.train_set.bundle_items(query)), 0)

        model = tiny_ddbc().fit(eval_method.train_set)
        items = model.construct(query, [], 4)
        self.assertEqual(len(set(items.tolist())), 4)
        self.assertEqual(model.score(query).shape, (N_ITEMS,))

        test_result, _ = eval_method.evaluate(
            tiny_ddbc(), [Recall(k=2)], user_based=False
        )
        for name in ["Recall@2", "Set-F1", "Set-Jaccard"]:
            self.assertGreaterEqual(test_result.metric_avg_results[name], 0.0)
            self.assertLessEqual(test_result.metric_avg_results[name], 1.0 + 1e-9)

    def test_evaluation(self):
        test_result, val_result = self.eval_method.evaluate(
            tiny_ddbc(), [Recall(k=2)], user_based=False
        )

        for result in (test_result, val_result):
            for name in ["Recall@2", "Set-F1", "Set-Jaccard"]:
                self.assertGreaterEqual(result.metric_avg_results[name], 0.0)
                self.assertLessEqual(result.metric_avg_results[name], 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
