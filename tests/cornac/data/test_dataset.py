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

import numpy as np
import numpy.testing as npt

from cornac.data import (
    BasketDataset,
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


if __name__ == "__main__":
    unittest.main()
