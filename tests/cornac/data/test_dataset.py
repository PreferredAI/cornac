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
from collections import OrderedDict

import numpy as np
import numpy.testing as npt

from cornac.data import Reader
from cornac.data import Dataset


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.triplet_data = Reader().read('./tests/data.txt')

    def test_init(self):
        train_set = Dataset.from_uir(self.triplet_data,
                                     global_uid_map=OrderedDict(),
                                     global_iid_map=OrderedDict())

        self.assertSequenceEqual(train_set.matrix.shape, (10, 10))
        self.assertEqual(train_set.min_rating, 3)
        self.assertEqual(train_set.max_rating, 5)

        self.assertEqual(int(train_set.global_mean), int((3 * 2 + 4 * 7 + 5) / 10))

        self.assertEqual(train_set.num_users, 10)
        self.assertEqual(train_set.num_items, 10)

        self.assertFalse(train_set.is_unk_user(7))
        self.assertTrue(train_set.is_unk_user(13))

        self.assertFalse(train_set.is_unk_item(3))
        self.assertTrue(train_set.is_unk_item(16))

        self.assertEqual(train_set.uid_map['768'], 1)
        self.assertEqual(train_set.iid_map['195'], 7)

        self.assertSequenceEqual(list(train_set.user_indices), range(10))
        self.assertListEqual(list(train_set.user_ids),
                             ['76', '768', '642', '930', '329', '633', '716', '871', '543', '754'])

        self.assertSequenceEqual(list(train_set.item_indices), range(10))
        self.assertListEqual(list(train_set.item_ids),
                             ['93', '257', '795', '709', '705', '226', '478', '195', '737', '282'])

    def test_idx_iter(self):
        train_set = Dataset.from_uir(self.triplet_data)

        ids = [batch_ids for batch_ids in train_set.idx_iter(idx_range=10, batch_size=1, shuffle=False)]
        npt.assert_array_equal(ids, np.arange(10).reshape(10, 1))

        ids = [batch_ids for batch_ids in train_set.idx_iter(idx_range=10, batch_size=1, shuffle=True)]
        npt.assert_raises(AssertionError, npt.assert_array_equal, ids, np.arange(10).reshape(10, 1))

    def test_uir_iter(self):
        train_set = Dataset.from_uir(self.triplet_data,
                                     global_uid_map=OrderedDict(),
                                     global_iid_map=OrderedDict())

        users = [batch_users for batch_users, _, _ in train_set.uir_iter()]
        self.assertSequenceEqual(users, range(10))

        items = [batch_items for _, batch_items, _ in train_set.uir_iter()]
        self.assertSequenceEqual(items, range(10))

        ratings = [batch_ratings for _, _, batch_ratings in train_set.uir_iter()]
        self.assertListEqual(ratings, [4, 4, 4, 4, 3, 4, 4, 5, 3, 4])

        ratings = [batch_ratings for _, _, batch_ratings in train_set.uir_iter(batch_size=5, num_zeros=1)]
        self.assertListEqual(ratings[0].tolist(), [4, 4, 4, 4, 3, 0, 0, 0, 0, 0])
        self.assertListEqual(ratings[1].tolist(), [4, 4, 5, 3, 4, 0, 0, 0, 0, 0])

    def test_uij_iter(self):
        train_set = Dataset.from_uir(self.triplet_data,
                                     global_uid_map=OrderedDict(),
                                     global_iid_map=OrderedDict(),
                                     seed=123)

        users = [batch_users for batch_users, _, _ in train_set.uij_iter()]
        self.assertSequenceEqual(users, range(10))

        pos_items = [batch_pos_items for _, batch_pos_items, _ in train_set.uij_iter()]
        self.assertSequenceEqual(pos_items, range(10))

        neg_items = [batch_neg_items for _, _, batch_neg_items in train_set.uij_iter()]
        self.assertRaises(AssertionError, self.assertSequenceEqual, neg_items, range(10))

        neg_items = [batch_neg_items for _, _, batch_neg_items in train_set.uij_iter(neg_sampling='popularity')]
        self.assertRaises(AssertionError, self.assertSequenceEqual, neg_items, range(10))

        try:
            for _ in train_set.uij_iter(neg_sampling='bla'): continue
        except ValueError:
            assert True


    def test_user_iter(self):
        train_set = Dataset.from_uir(self.triplet_data,
                                     global_uid_map=OrderedDict(),
                                     global_iid_map=OrderedDict())

        npt.assert_array_equal(np.arange(10).reshape(10, 1),
                               [u for u in train_set.user_iter()])
        self.assertRaises(AssertionError, npt.assert_array_equal,
                          np.arange(10).reshape(10, 1),
                          [u for u in train_set.user_iter(shuffle=True)])

    def test_item_iter(self):
        train_set = Dataset.from_uir(self.triplet_data,
                                     global_uid_map=OrderedDict(),
                                     global_iid_map=OrderedDict())

        npt.assert_array_equal(np.arange(10).reshape(10, 1),
                               [i for i in train_set.item_iter()])
        self.assertRaises(AssertionError, npt.assert_array_equal,
                          np.arange(10).reshape(10, 1),
                          [i for i in train_set.item_iter(shuffle=True)])

    def test_uir_tuple(self):
        train_set = Dataset.from_uir(self.triplet_data,
                                     global_uid_map=None,
                                     global_iid_map=None)

        self.assertEqual(len(train_set.uir_tuple), 3)
        self.assertEqual(len(train_set.uir_tuple[0]), 10)

        try:
            train_set.uir_tuple = ([], [])
        except ValueError:
            assert True

        self.assertEqual(train_set.num_batches(batch_size=5), 2)

    def test_matrix(self):
        from scipy.sparse import csr_matrix, csc_matrix, dok_matrix

        train_set = Dataset.from_uir(self.triplet_data,
                                     global_uid_map=None,
                                     global_iid_map=None)

        self.assertTrue(isinstance(train_set.matrix, csr_matrix))
        self.assertEqual(train_set.csr_matrix[0, 0], 4)
        self.assertTrue(train_set.csr_matrix.has_sorted_indices)

        self.assertTrue(isinstance(train_set.csc_matrix, csc_matrix))
        self.assertEqual(train_set.csc_matrix[4, 4], 3)

        self.assertTrue(isinstance(train_set.dok_matrix, dok_matrix))
        self.assertEqual(train_set.dok_matrix[7, 7], 5)

    def test_user_data(self):
        train_set = Dataset.from_uir(self.triplet_data,
                                     global_uid_map=None,
                                     global_iid_map=None)

        self.assertEqual(len(train_set.user_data), 10)
        self.assertListEqual(train_set.user_data[0][0], [0])
        self.assertListEqual(train_set.user_data[0][1], [4.0])

    def test_item_data(self):
        train_set = Dataset.from_uir(self.triplet_data,
                                     global_uid_map=None,
                                     global_iid_map=None)

        self.assertEqual(len(train_set.item_data), 10)
        self.assertListEqual(train_set.user_data[0][0], [0])
        self.assertListEqual(train_set.user_data[0][1], [4.0])


if __name__ == '__main__':
    unittest.main()
