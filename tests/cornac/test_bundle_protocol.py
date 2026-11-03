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
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from cornac.data import BundleDataset
from cornac.datasets import bundle
from cornac.eval_methods import BundleEvaluation
from cornac.metrics import Recall, NDCG
from cornac.models import BundleRecommender


UB = [("0", "0", 1), ("1", "1", 1)]
BI = [("0", "0"), ("1", "1"), ("2", "0")]
UI = [("0", "0", 1), ("1", "1", 1), ("2", "2", 1)]
IDS = dict(user_ids=["0", "1", "2", "3"], bundle_ids=["0", "1", "2"],
           constituent_item_ids=["0", "1", "2", "3"])


class OrderedScore(BundleRecommender):
    def __init__(self):
        super().__init__("OrderedScore")
    def score(self, user_idx, item_idx=None):
        scores = np.array([0., 3., 2., 1.])
        return scores if item_idx is None else scores[item_idx]


class TestBundleProtocol(unittest.TestCase):
    def test_explicit_catalog_keeps_ui_only_users_and_unused_ids(self):
        data = BundleDataset.build(UB, BI, UI, **IDS)
        self.assertEqual(data.num_users, 4)
        self.assertEqual(data.user_item_matrix.nnz, 3)
        self.assertEqual(data.user_item_matrix.shape, (4, 4))
        self.assertEqual(data.bundle_item_matrix.shape, (3, 4))
        self.assertEqual(data.uid_map["3"], 3)
        self.assertEqual(data.constituent_iid_map["3"], 3)
        self.assertEqual(BundleDataset.build(UB, BI, UI).user_item_matrix.nnz, 2)

    def test_supplied_catalog_order_controls_indices(self):
        ids = {key: list(reversed(values)) for key, values in IDS.items()}
        data = BundleDataset.build(UB, BI, UI, **ids)
        self.assertEqual(data.uid_map["0"], 3)
        self.assertEqual(data.bid_map["0"], 2)
        self.assertEqual(data.constituent_iid_map["0"], 3)
        self.assertEqual(data.matrix[3, 2], 1)
        self.assertEqual(data.bundle_item_matrix[2, 3], 1)

    def test_evaluation_preserves_full_catalog(self):
        method = BundleEvaluation.from_splits(UB, [("0", "2", 1)],
            val_data=[("1", "2", 1)], bundle_item_data=BI, user_item_data=UI, **IDS)
        for data in (method.train_set, method.test_set, method.val_set):
            self.assertEqual(data.user_item_matrix.nnz, 3)
            self.assertEqual(data.user_item_matrix.shape, (4, 4))
            self.assertEqual(data.bundle_item_matrix.shape, (3, 4))
            self.assertEqual(data.uid_map["3"], 3)

    def test_invalid_catalogs(self):
        for updates in (dict(user_ids=["0", "0"]), dict(user_ids=["0"]),
                        dict(bundle_ids=["0", "1"]), dict(bundle_ids=["0", "1", "2", "3"]),
                        dict(constituent_item_ids=["0"])):
            ids = dict(IDS, **updates)
            with self.assertRaises(ValueError):
                BundleDataset.build(UB, BI, UI, **ids)
        with self.assertRaisesRegex(ValueError, "undeclared user"):
            BundleDataset.build(UB, BI, UI + [("bad", "0", 1)], **IDS)
        with self.assertRaises(ValueError):
            BundleEvaluation.from_splits(UB, [("bad", "2", 1)],
                bundle_item_data=BI, user_item_data=UI, **IDS)

    def test_reference_train_only_test_mask(self):
        membership = [(str(index), str(index)) for index in range(4)]
        for mode, recall in (("train_and_validation", 1), ("train", 0)):
            method = BundleEvaluation.from_splits([("u", "0", 1)], [("u", "2", 1)],
                val_data=[("u", "1", 1)], bundle_item_data=membership, test_mask=mode)
            model = OrderedScore()
            result, validation = method.evaluate(model, [Recall(k=1), NDCG(k=1)], True)
            self.assertEqual(result.metric_avg_results["Recall@1"], recall)
            self.assertEqual(result.metric_avg_results["NDCG@1"], recall)
            self.assertEqual(validation.metric_avg_results["Recall@1"], 1)
            self.assertIs(model.val_set, method.val_set)


class TestBundleLoaders(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        (self.path / "pairs").write_text("0\t1\n2\t3\n")
        (self.path / "metadata").write_text("4\t3\t5\n")
    def tearDown(self):
        self.directory.cleanup()

    def test_feedback_alias_cache_and_reader(self):
        with patch.object(bundle, "cache", return_value=str(self.path / "pairs")) as cache:
            rows = bundle.load_feedback("yOuShU", "validation", cache_dir="/tmp/test-cache")
            self.assertEqual(rows, [("0", "1", 1.), ("2", "3", 1.)])
            self.assertEqual(cache.call_args.kwargs["relative_path"], "datasets/Youshu/user_bundle_tune.txt")
            self.assertEqual(cache.call_args.kwargs["cache_dir"], "/tmp/test-cache")
            reader = Mock()
            reader.read.return_value = [("filtered", "item", 1.)]
            self.assertEqual(bundle.load_user_item("NETEASE", reader), reader.read.return_value)
            reader.read.assert_called_once_with(str(self.path / "pairs"), fmt="UI", sep="\t")

    def test_membership_and_ids(self):
        with patch.object(bundle, "cache", return_value=str(self.path / "pairs")):
            self.assertEqual(bundle.load_bundle_item("ifashion"), [("0", "1"), ("2", "3")])
        with patch.object(bundle, "cache", return_value=str(self.path / "metadata")):
            ids = bundle.load_ids("youshu")
            self.assertEqual(list(ids), ["user_ids", "bundle_ids", "constituent_item_ids"])
            self.assertEqual(ids["user_ids"], ["0", "1", "2", "3"])
            self.assertEqual(ids["constituent_item_ids"], ["0", "1", "2", "3", "4"])

    def test_validation_before_network(self):
        with patch.object(bundle, "cache") as cache:
            for fn in (bundle.load_feedback, bundle.load_user_item, bundle.load_bundle_item, bundle.load_ids):
                with self.assertRaises(ValueError):
                    fn("unknown")
            with self.assertRaises(ValueError):
                bundle.load_feedback(split="invalid")
            cache.assert_not_called()

    def test_spotify_feedback_and_cache(self):
        (self.path / "spotify").write_text("10\t2\t1.0\n2\t0\t1.0\n")
        with patch.object(bundle, "cache", return_value=str(self.path / "spotify")) as cache:
            self.assertEqual(bundle.load_feedback("sPoTiFy", "tune", cache_dir="/tmp/test-cache"),
                             [("10", "2", 1.0), ("2", "0", 1.0)])
            self.assertEqual(cache.call_args.kwargs, dict(
                url=bundle.SPOTIFY_URL + "validation.txt",
                relative_path="datasets/AttList/Spotify/validation.txt", cache_dir="/tmp/test-cache"))
            reader = Mock()
            bundle.load_feedback("Spotify", "test", reader=reader)
            reader.read.assert_called_once_with(str(self.path / "spotify"), fmt="UIR", sep="\t")

    def test_spotify_membership_catalog_and_no_ui(self):
        (self.path / "listItem_spotify.txt").write_text("2\t[10, 1, 10]\n0\t[2, 1]\n")
        # Held-out list IDs in userList must never become catalog bundles or features.
        (self.path / "userList_spotify.txt").write_text("10\t[999]\n2\t[998]\n0\t[]\n")
        with patch.object(bundle, "cache", side_effect=lambda **kw: str(self.path / kw["url"].split("/")[-1])):
            self.assertEqual(bundle.load_bundle_item("Spotify"),
                             [("2", "10"), ("2", "1"), ("0", "2"), ("0", "1")])
            ids = bundle.load_ids("SPOTIFY")
            self.assertEqual(ids, dict(user_ids=["0", "2", "10"], bundle_ids=["0", "2"],
                                       constituent_item_ids=["1", "2", "10"]))
            method = BundleEvaluation.from_splits(
                train_data=[("0", "0", 1), ("2", "2", 1)],
                val_data=[("10", "0", 1)], test_data=[("10", "2", 1)],
                bundle_item_data=bundle.load_bundle_item("Spotify"), **ids)
            self.assertIsNone(method.train_set.user_item_matrix)
            self.assertEqual(list(method.train_set.constituent_iid_map.items()),
                             [("1", 0), ("2", 1), ("10", 2)])
        with patch.object(bundle, "cache") as cache:
            with self.assertRaisesRegex(ValueError, "no independent user-item"):
                bundle.load_user_item("Spotify")
            cache.assert_not_called()
        (self.path / "listItem_spotify.txt").write_text("0\t__import__('os')\n")
        with patch.object(bundle, "cache", return_value=str(self.path / "listItem_spotify.txt")):
            with self.assertRaises(ValueError):
                bundle.load_bundle_item("Spotify")
