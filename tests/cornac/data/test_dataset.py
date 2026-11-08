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
import warnings
from collections import OrderedDict

import numpy as np
import numpy.testing as npt

from cornac.data import (
    BasketDataset,
    BundleConstructionDataset,
    BundleDataset,
    Dataset,
    PurchaseViewDataset,
    SequentialDataset,
    Reader,
)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.triplet_data = Reader().read("./tests/data.txt")
        self.uirt_data = Reader().read("./tests/data.txt", fmt="UIRT")

    def test_init(self):
        train_set = Dataset.from_uir(self.triplet_data)

        self.assertSequenceEqual(train_set.matrix.shape, (10, 10))
        self.assertEqual(train_set.min_rating, 3)
        self.assertEqual(train_set.max_rating, 5)

        self.assertEqual(int(train_set.global_mean), int((3 * 2 + 4 * 7 + 5) / 10))

        self.assertEqual(train_set.num_users, 10)
        self.assertEqual(train_set.num_items, 10)

        self.assertEqual(train_set.uid_map["768"], 1)
        self.assertEqual(train_set.iid_map["195"], 7)

        self.assertSetEqual(
            set(train_set.user_ids),
            set(["76", "768", "642", "930", "329", "633", "716", "871", "543", "754"]),
        )

        self.assertSetEqual(
            set(train_set.item_ids),
            set(["93", "257", "795", "709", "705", "226", "478", "195", "737", "282"]),
        )

    def test_from_uirt(self):
        train_set = Dataset.from_uirt(self.uirt_data)

        self.assertTrue(len(train_set.timestamps) == 10)

    def test_exclude_unknowns_empty_error(self):
        try:
            Dataset.build(self.triplet_data, exclude_unknowns=True)
        except ValueError:
            assert True

    def test_idx_iter(self):
        train_set = Dataset.from_uir(self.triplet_data)

        ids = [
            batch_ids
            for batch_ids in train_set.idx_iter(
                idx_range=10, batch_size=1, shuffle=False
            )
        ]
        npt.assert_array_equal(ids, np.arange(10).reshape(10, 1))

        ids = [
            batch_ids
            for batch_ids in train_set.idx_iter(
                idx_range=10, batch_size=1, shuffle=True
            )
        ]
        npt.assert_raises(
            AssertionError, npt.assert_array_equal, ids, np.arange(10).reshape(10, 1)
        )

    def test_uir_iter(self):
        train_set = Dataset.from_uir(self.triplet_data)

        users = [batch_users for batch_users, _, _ in train_set.uir_iter()]
        self.assertSequenceEqual(users, range(10))

        items = [batch_items for _, batch_items, _ in train_set.uir_iter()]
        self.assertSequenceEqual(items, range(10))

        ratings = [batch_ratings for _, _, batch_ratings in train_set.uir_iter()]
        self.assertListEqual(ratings, [4, 4, 4, 4, 3, 4, 4, 5, 3, 4])

        ratings = [
            batch_ratings for _, _, batch_ratings in train_set.uir_iter(binary=True)
        ]
        self.assertListEqual(ratings, [1] * 10)

        ratings = [
            batch_ratings
            for _, _, batch_ratings in train_set.uir_iter(batch_size=5, num_zeros=1)
        ]
        self.assertListEqual(ratings[0].tolist(), [4, 4, 4, 4, 3, 0, 0, 0, 0, 0])
        self.assertListEqual(ratings[1].tolist(), [4, 4, 5, 3, 4, 0, 0, 0, 0, 0])

    def test_uij_iter(self):
        train_set = Dataset.from_uir(self.triplet_data, seed=123)

        users = [batch_users for batch_users, _, _ in train_set.uij_iter()]
        self.assertSequenceEqual(users, range(10))

        pos_items = [batch_pos_items for _, batch_pos_items, _ in train_set.uij_iter()]
        self.assertSequenceEqual(pos_items, range(10))

        neg_items = [batch_neg_items for _, _, batch_neg_items in train_set.uij_iter()]
        self.assertRaises(
            AssertionError, self.assertSequenceEqual, neg_items, range(10)
        )

        neg_items = [
            batch_neg_items
            for _, _, batch_neg_items in train_set.uij_iter(neg_sampling="popularity")
        ]
        self.assertRaises(
            AssertionError, self.assertSequenceEqual, neg_items, range(10)
        )

        try:
            for _ in train_set.uij_iter(neg_sampling="bla"):
                continue
        except ValueError:
            assert True

    def test_user_iter(self):
        train_set = Dataset.from_uir(self.triplet_data)

        npt.assert_array_equal(
            np.arange(10).reshape(10, 1), [u for u in train_set.user_iter()]
        )
        self.assertRaises(
            AssertionError,
            npt.assert_array_equal,
            np.arange(10).reshape(10, 1),
            [u for u in train_set.user_iter(shuffle=True)],
        )

    def test_item_iter(self):
        train_set = Dataset.from_uir(self.triplet_data)

        npt.assert_array_equal(
            np.arange(10).reshape(10, 1), [i for i in train_set.item_iter()]
        )
        self.assertRaises(
            AssertionError,
            npt.assert_array_equal,
            np.arange(10).reshape(10, 1),
            [i for i in train_set.item_iter(shuffle=True)],
        )

    def test_uir_tuple(self):
        train_set = Dataset.from_uir(self.triplet_data)

        self.assertEqual(len(train_set.uir_tuple), 3)
        self.assertEqual(len(train_set.uir_tuple[0]), 10)
        self.assertEqual(train_set.num_batches(batch_size=5), 2)

    def test_matrix(self):
        from scipy.sparse import csc_matrix, csr_matrix, dok_matrix

        train_set = Dataset.from_uir(self.triplet_data)

        self.assertTrue(isinstance(train_set.matrix, csr_matrix))
        self.assertEqual(train_set.csr_matrix[0, 0], 4)
        self.assertTrue(train_set.csr_matrix.has_sorted_indices)

        self.assertTrue(isinstance(train_set.csc_matrix, csc_matrix))
        self.assertEqual(train_set.csc_matrix[4, 4], 3)

        self.assertTrue(isinstance(train_set.dok_matrix, dok_matrix))
        self.assertEqual(train_set.dok_matrix[7, 7], 5)

    def test_user_data(self):
        train_set = Dataset.from_uir(self.triplet_data)

        self.assertEqual(len(train_set.user_data), 10)
        self.assertListEqual(train_set.user_data[0][0], [0])
        self.assertListEqual(train_set.user_data[0][1], [4.0])

    def test_item_data(self):
        train_set = Dataset.from_uir(self.triplet_data)

        self.assertEqual(len(train_set.item_data), 10)
        self.assertListEqual(train_set.item_data[0][0], [0])
        self.assertListEqual(train_set.item_data[0][1], [4.0])

    def test_chrono_user_data(self):
        zero_data = []
        for idx in range(len(self.triplet_data)):
            u = self.triplet_data[idx][0]
            i = self.triplet_data[-1 - idx][1]
            zero_data.append((u, i, 1.0, 0))
        train_set = Dataset.from_uirt(self.uirt_data + zero_data)

        self.assertEqual(len(train_set.chrono_user_data), 10)
        self.assertListEqual(train_set.chrono_user_data[0][1], [1.0, 4.0])
        self.assertListEqual(train_set.chrono_user_data[0][2], [0, 882606572])

        try:
            Dataset.from_uir(self.triplet_data).chrono_user_data
        except ValueError:
            assert True

    def test_chrono_item_data(self):
        zero_data = []
        for idx in range(len(self.triplet_data)):
            u = self.triplet_data[idx][0]
            i = self.triplet_data[-1 - idx][1]
            zero_data.append((u, i, 1.0, 0))
        train_set = Dataset.from_uirt(self.uirt_data + zero_data)

        self.assertEqual(len(train_set.chrono_item_data), 10)
        self.assertListEqual(train_set.chrono_item_data[0][1], [1.0, 4.0])
        self.assertListEqual(train_set.chrono_item_data[0][2], [0, 882606572])

        try:
            Dataset.from_uir(self.triplet_data).chrono_item_data
        except ValueError:
            assert True


class TestBasketDataset(unittest.TestCase):
    def setUp(self):
        self.basket_data = Reader().read("./tests/basket.txt", fmt="UBITJson")

    def test_init(self):
        train_set = BasketDataset.from_ubi(self.basket_data)

        self.assertEqual(train_set.num_baskets, 25)
        self.assertEqual(train_set.max_basket_size, 3)
        self.assertEqual(train_set.min_basket_size, 1)

        self.assertEqual(train_set.num_users, 10)
        self.assertEqual(train_set.num_items, 7)

        self.assertEqual(train_set.uid_map["1"], 0)
        self.assertEqual(train_set.bid_map["1"], 0)
        self.assertEqual(train_set.iid_map["1"], 0)

        self.assertSetEqual(
            set(train_set.user_ids),
            set(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]),
        )

        self.assertSetEqual(
            set(train_set.item_ids),
            set(["1", "2", "3", "4", "5", "6", "7"]),
        )


class TestSequentialDataset(unittest.TestCase):
    def setUp(self):
        self.sequential_data = Reader().read("./tests/sequence.txt", fmt="USIT", sep=" ")

    def test_init(self):
        train_set = SequentialDataset.from_usit(self.sequential_data)

        self.assertEqual(train_set.num_sessions, 16)
        self.assertEqual(train_set.max_session_size, 6)
        self.assertEqual(train_set.min_session_size, 2)

        self.assertEqual(train_set.num_users, 5)
        self.assertEqual(train_set.num_items, 9)

        self.assertEqual(train_set.uid_map["1"], 0)
        self.assertEqual(train_set.sid_map["1"], 0)
        self.assertEqual(train_set.iid_map["1"], 0)

        self.assertSetEqual(
            set(train_set.user_ids),
            set(["1", "2", "3", "4", "5"]),
        )

        self.assertSetEqual(
            set(train_set.item_ids),
            set(["1", "2", "3", "4", "5", "6", "7", "8", "9"]),
        )


class TestSequentialDatasetSortOrder(unittest.TestCase):
    def _assert_no_warning(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SequentialDataset.from_sit(data)

    def test_shuffled_within_session_warns(self):
        data = [("s1", "a", 3), ("s1", "b", 1), ("s1", "c", 2)]
        with self.assertWarns(UserWarning):
            SequentialDataset.from_sit(data)

    def test_per_session_sorted_no_warning(self):
        data = [
            ("s1", "a", 1),
            ("s1", "b", 2),
            ("s2", "c", 1),
            ("s2", "d", 2),
        ]
        self._assert_no_warning(data)

    def test_interleaved_sessions_sorted_no_warning(self):
        # Rows of session A and B alternate; each session is internally sorted.
        data = [
            ("A", "x", 1),
            ("B", "p", 1),
            ("A", "y", 2),
            ("B", "q", 2),
        ]
        self._assert_no_warning(data)

    def test_noncontiguous_cross_block_disorder_warns(self):
        # Session A appears in two blocks; the later A block is earlier in time.
        data = [
            ("A", "x", 5),
            ("A", "y", 6),
            ("B", "p", 1),
            ("A", "z", 2),
        ]
        with self.assertWarns(UserWarning):
            SequentialDataset.from_sit(data)

    def test_tied_timestamps_no_warning(self):
        data = [("A", "x", 1), ("A", "y", 1), ("A", "z", 1)]
        self._assert_no_warning(data)

    def test_single_row_no_warning(self):
        self._assert_no_warning([("A", "x", 1)])

    def test_empty_data_no_sort_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(ValueError):
                SequentialDataset.from_sit([])


class TestPurchaseViewDataset(unittest.TestCase):
    def test_build_extends_id_space(self):
        # User "u2" and item "i3" appear only in the view stream; they must
        # end up in the shared id maps rather than being silently dropped.
        purchase_data = [("u1", "i1", 1.0), ("u1", "i2", 1.0)]
        view_data = [("u2", "i3", 1.0), ("u1", "i2", 1.0)]

        dataset = PurchaseViewDataset.build(purchase_data, view_data)

        self.assertIn("u2", dataset.uid_map)
        self.assertIn("i3", dataset.iid_map)
        self.assertEqual(dataset.num_users, 2)
        self.assertEqual(dataset.num_items, 3)

    def test_view_matrix_invariants(self):
        purchase_data = [("u1", "i1", 1.0), ("u2", "i2", 1.0)]
        view_data = [("u1", "i2", 1.0), ("u2", "i1", 1.0)]

        dataset = PurchaseViewDataset.build(purchase_data, view_data)

        self.assertEqual(dataset.view_matrix.format, "csr")
        self.assertTrue(dataset.view_matrix.has_sorted_indices)
        self.assertEqual(
            dataset.view_matrix.shape, (dataset.num_users, dataset.num_items)
        )

    def test_is_compatible_with_dataset_api(self):
        # A PurchaseViewDataset should be indistinguishable from a plain
        # Dataset when only the primary (purchase) feedback is consumed —
        # this is what lets vanilla BPR train on it unchanged.
        purchase_data = [("u1", "i1", 1.0), ("u2", "i2", 1.0)]
        view_data = [("u1", "i2", 1.0)]

        pv = PurchaseViewDataset.build(purchase_data, view_data)
        plain = Dataset.from_uir(purchase_data)

        self.assertIsInstance(pv, Dataset)
        self.assertEqual(pv.num_users, plain.num_users)
        self.assertEqual(pv.num_items, plain.num_items)
        self.assertEqual(pv.matrix.nnz, plain.matrix.nnz)
        npt.assert_array_equal(pv.uir_tuple[0], plain.uir_tuple[0])
        npt.assert_array_equal(pv.uir_tuple[1], plain.uir_tuple[1])
        npt.assert_array_equal(pv.uir_tuple[2], plain.uir_tuple[2])

    def test_view_excludes_purchase_overlap(self):
        # Paper defines v as "viewed but not purchased"; view entries that
        # collide with purchase entries must be dropped so the sampler can
        # never draw v == i.
        purchase_data = [("u1", "i1", 1.0), ("u1", "i2", 1.0)]
        view_data = [("u1", "i1", 1.0), ("u1", "i3", 1.0)]  # i1 overlaps

        dataset = PurchaseViewDataset.build(purchase_data, view_data)

        u1 = dataset.uid_map["u1"]
        i1 = dataset.iid_map["i1"]
        i3 = dataset.iid_map["i3"]
        self.assertEqual(dataset.view_matrix[u1, i1], 0.0)
        self.assertNotEqual(dataset.view_matrix[u1, i3], 0.0)

    def test_attach_view_drops_unknown_entries(self):
        # attach_view aligns to an existing purchase dataset's ID space:
        # view rows referencing users/items not in that space are filtered out.
        purchase = Dataset.from_uir([("u1", "i1", 1.0), ("u2", "i2", 1.0)])
        view_data = [
            ("u1", "i2", 1.0),       # both known — kept
            ("u_unknown", "i1", 1.0),  # unknown user — dropped
            ("u2", "i_unknown", 1.0),  # unknown item — dropped
        ]

        dataset = PurchaseViewDataset.attach_view(purchase, view_data)

        self.assertEqual(dataset.num_users, purchase.num_users)
        self.assertEqual(dataset.num_items, purchase.num_items)
        self.assertEqual(dataset.view_matrix.nnz, 1)


BUNDLE_CATALOG = [("b1", "b1"), ("b1", "i2"), ("b2", "i2"), ("b3", "i3")]
BUNDLE_TRAIN = [("u1", "b1", 1), ("u2", "b2", 1)]
CATALOG_UB = [("0", "0", 1), ("1", "1", 1)]
CATALOG_BI = [("0", "0"), ("1", "1"), ("2", "0")]
CATALOG_UI = [("0", "0", 1), ("1", "1", 1), ("2", "2", 1)]
CATALOG_IDS = dict(
    user_ids=["0", "1", "2", "3"],
    bundle_ids=["0", "1", "2"],
    constituent_item_ids=["0", "1", "2", "3"],
)


class TestBundleDataset(unittest.TestCase):
    def test_catalog_and_separate_ids(self):
        data = BundleDataset.build(BUNDLE_TRAIN, BUNDLE_CATALOG + [BUNDLE_CATALOG[0]])
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
        npt.assert_array_equal(
            data.bundle_item_matrix.toarray(), [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        )

    def test_independent_side_feedback(self):
        side = [("u1", "i4", 2), ("u1", "i4", 3), ("other", "ignored", 1)]
        data = BundleDataset.build(iter(BUNDLE_TRAIN), iter(BUNDLE_CATALOG), iter(side))
        self.assertEqual(data.user_item_matrix.shape, (2, 4))
        self.assertEqual(data.bundle_item_matrix.shape, (3, 4))
        self.assertEqual(data.user_item_matrix[0, data.constituent_iid_map["i4"]], 2)
        self.assertNotIn("ignored", data.constituent_iid_map)
        self.assertEqual(data.user_item_matrix.nnz, 1)

    def test_empty_side(self):
        for side in ([], [("other", "ignored", 1)]):
            data = BundleDataset.build(BUNDLE_TRAIN, BUNDLE_CATALOG, side)
            self.assertEqual(data.user_item_matrix.shape, (2, 3))
            self.assertEqual(data.user_item_matrix.nnz, 0)

    def test_invalid_catalog(self):
        with self.assertRaisesRegex(ValueError, "nonempty catalog"):
            BundleDataset.build(BUNDLE_TRAIN, [])
        with self.assertRaisesRegex(ValueError, "absent from the catalog"):
            BundleDataset.build(BUNDLE_TRAIN + [("u1", "missing", 1)], BUNDLE_CATALOG)

    def test_shared_constituent_mapping_is_fixed(self):
        data = BundleDataset.build(BUNDLE_TRAIN, BUNDLE_CATALOG)
        before = dict(data.constituent_iid_map)
        with self.assertRaisesRegex(ValueError, "shared mapping"):
            BundleDataset.from_dataset(
                data,
                BUNDLE_CATALOG,
                [("u1", "new_item", 1)],
                constituent_iid_map=data.constituent_iid_map,
            )
        self.assertEqual(dict(data.constituent_iid_map), before)
        self.assertEqual(data.bundle_item_matrix.shape, (3, 3))

    def test_timestamp_and_duplicate_feedback(self):
        with self.assertWarns(UserWarning):
            data = BundleDataset.build(
                [("u1", "b1", 1, 10), ("u1", "b1", 2, 20)],
                BUNDLE_CATALOG,
                fmt="UIRT",
            )
        npt.assert_array_equal(data.timestamps, [10])
        self.assertEqual(data.num_ratings, 1)

    def test_explicit_catalog_keeps_ui_only_users_and_unused_ids(self):
        data = BundleDataset.build(CATALOG_UB, CATALOG_BI, CATALOG_UI, **CATALOG_IDS)
        self.assertEqual(data.num_users, 4)
        self.assertEqual(data.user_item_matrix.nnz, 3)
        self.assertEqual(data.user_item_matrix.shape, (4, 4))
        self.assertEqual(data.bundle_item_matrix.shape, (3, 4))
        self.assertEqual(data.uid_map["3"], 3)
        self.assertEqual(data.constituent_iid_map["3"], 3)
        self.assertEqual(
            BundleDataset.build(
                CATALOG_UB, CATALOG_BI, CATALOG_UI
            ).user_item_matrix.nnz,
            2,
        )

    def test_supplied_catalog_order_controls_indices(self):
        ids = {key: list(reversed(values)) for key, values in CATALOG_IDS.items()}
        data = BundleDataset.build(CATALOG_UB, CATALOG_BI, CATALOG_UI, **ids)
        self.assertEqual(data.uid_map["0"], 3)
        self.assertEqual(data.bid_map["0"], 2)
        self.assertEqual(data.constituent_iid_map["0"], 3)
        self.assertEqual(data.matrix[3, 2], 1)
        self.assertEqual(data.bundle_item_matrix[2, 3], 1)

    def test_invalid_catalogs(self):
        for updates in (
            dict(user_ids=["0", "0"]),
            dict(user_ids=["0"]),
            dict(bundle_ids=["0", "1"]),
            dict(bundle_ids=["0", "1", "2", "3"]),
            dict(constituent_item_ids=["0"]),
        ):
            ids = dict(CATALOG_IDS, **updates)
            with self.assertRaises(ValueError):
                BundleDataset.build(CATALOG_UB, CATALOG_BI, CATALOG_UI, **ids)
        with self.assertRaisesRegex(ValueError, "undeclared user"):
            BundleDataset.build(
                CATALOG_UB, CATALOG_BI, CATALOG_UI + [("bad", "0", 1)], **CATALOG_IDS
            )


class TestBundleConstructionDataset(unittest.TestCase):
    def test_bi_and_bit(self):
        data = BundleConstructionDataset.build(
            [("b1", "i1"), ("b1", "i2"), ("b2", "i2")]
        )
        self.assertEqual((data.num_bundles, data.num_items, data.num_owners), (2, 2, 0))
        self.assertIs(data.bid_map, data.uid_map)
        self.assertEqual(data.bundle_ids, ["b1", "b2"])
        npt.assert_array_equal(data.owner_indices, [-1, -1])
        npt.assert_array_equal(data.matrix.toarray(), [[1, 1], [0, 1]])
        self.assertIsNone(data.timestamps)
        self.assertIsNone(data.user_bundle_matrix)
        self.assertIsNone(data.user_item_matrix)
        self.assertEqual(len(data.partial_bundles), 0)
        timed = BundleConstructionDataset.build(
            [("b1", "i1", 5), ("b1", "i2", 3)], fmt="BIT"
        )
        npt.assert_array_equal(timed.timestamps, [5, 3])
        npt.assert_array_equal(timed.bundle_items(0), [1, 0])

    def test_ubi_and_ubit_owners(self):
        owners = OrderedDict([("u0", 0)])
        data = BundleConstructionDataset.build(
            [("u2", "b1", "i1"), ("u1", "b2", "i1"), ("u2", "b3", "i2")],
            fmt="UBI",
            global_owner_map=owners,
        )
        self.assertIs(data.owner_map, owners)
        self.assertEqual(list(owners), ["u0", "u2", "u1"])
        npt.assert_array_equal(data.owner_indices, [1, 2, 1])
        self.assertEqual(data.num_owners, 3)
        timed = BundleConstructionDataset.build(
            [("u1", "b1", "i1", 2), ("u1", "b1", "i2", 1)], fmt="UBIT"
        )
        npt.assert_array_equal(timed.owner_indices, [0])
        npt.assert_array_equal(timed.bundle_items(0), [1, 0])
        for fmt, rows in (
            ("UBI", [("u1", "b1", "i1"), ("u2", "b1", "i2")]),
            ("UBIT", [("u1", "b1", "i1", 1), ("u2", "b1", "i2", 2)]),
        ):
            with self.assertRaisesRegex(ValueError, "conflicting owners"):
                BundleConstructionDataset.build(rows, fmt=fmt)
        with self.assertRaises(ValueError):
            BundleConstructionDataset.build([("b1", "i1", 1.0)], fmt="UIR")

    def test_bundle_items_order(self):
        data = BundleConstructionDataset.build(
            [("b2", "i1"), ("b1", "i3"), ("b1", "i1"), ("b1", "i2")],
            global_uid_map=OrderedDict([("empty", 0)]),
        )
        npt.assert_array_equal(
            data.bundle_items(data.bid_map["b1"]), [1, 0, 2]
        )  # row order
        npt.assert_array_equal(data.bundle_items(data.bid_map["b2"]), [0])
        self.assertEqual(len(data.bundle_items(0)), 0)
        with self.assertWarns(UserWarning):  # duplicate (b1, i1) is removed
            timed = BundleConstructionDataset.build(
                [("b1", "i1", 2), ("b1", "i2", 1), ("b1", "i3", 2), ("b1", "i1", 0)],
                fmt="BIT",
            )
        npt.assert_array_equal(
            timed.bundle_items(0), [1, 0, 2]
        )  # stable timestamp order


if __name__ == "__main__":
    unittest.main()
