# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
import numpy as np
import numpy.testing as npt
from collections import OrderedDict

from cornac.data import Reader
from cornac.data import TrainSet
from cornac.data import MatrixTrainSet


class TestTrainSet(unittest.TestCase):

    def test_init(self):
        uid_map = OrderedDict([('a', 0), ('b', 1)])
        iid_map = OrderedDict([('x', 0), ('y', 1), ('z', 2)])

        train_set = TrainSet(uid_map, iid_map)

        self.assertEqual(train_set.num_users, 2)
        self.assertEqual(train_set.num_items, 3)

        self.assertTrue(train_set.is_unk_user(2))
        self.assertFalse(train_set.is_unk_user(1))

        self.assertTrue(train_set.is_unk_item(4))
        self.assertFalse(train_set.is_unk_item(2))

        self.assertEqual(train_set.get_uid('b'), 1)
        self.assertEqual(train_set.get_iid('y'), 1)

        self.assertListEqual(train_set.uid_list, list(uid_map.values()))
        self.assertListEqual(train_set.raw_uid_list, list(uid_map.keys()))

        self.assertListEqual(train_set.iid_list, list(iid_map.values()))
        self.assertListEqual(train_set.raw_iid_list, list(iid_map.keys()))

    def test_idx_iter(self):
        ids = [batch_ids for batch_ids in TrainSet.idx_iter(idx_range=10, batch_size=1, shuffle=False)]
        npt.assert_array_equal(ids, np.arange(10).reshape(10, 1))

        ids = [batch_ids for batch_ids in TrainSet.idx_iter(idx_range=10, batch_size=1, shuffle=True)]
        npt.assert_raises(AssertionError, npt.assert_array_equal, ids, np.arange(10).reshape(10, 1))


class TestMatrixTrainSet(unittest.TestCase):

    def setUp(self):
        self.triplet_data = Reader().read('./tests/data.txt')

    def test_init(self):
        train_set = MatrixTrainSet.from_uir(self.triplet_data,
                                            global_uid_map=OrderedDict(),
                                            global_iid_map=OrderedDict(),
                                            global_ui_set=set(),
                                            verbose=True)

        self.assertSequenceEqual(train_set.matrix.shape, (10, 10))
        self.assertEqual(train_set.min_rating, 3)
        self.assertEqual(train_set.max_rating, 5)

        self.assertEqual(int(train_set.global_mean), int((3 * 2 + 4 * 7 + 5) / 10))

        npt.assert_array_equal(train_set.item_ppl_rank, np.asarray([7, 9, 6, 5, 3, 2, 1, 0, 8, 4]))

        self.assertEqual(train_set.num_users, 10)
        self.assertEqual(train_set.num_items, 10)

        self.assertFalse(train_set.is_unk_user(7))
        self.assertTrue(train_set.is_unk_user(13))

        self.assertFalse(train_set.is_unk_item(3))
        self.assertTrue(train_set.is_unk_item(16))

        self.assertEqual(train_set.get_uid('768'), 1)
        self.assertEqual(train_set.get_iid('195'), 7)

        self.assertSequenceEqual(train_set.uid_list, range(10))
        self.assertListEqual(train_set.raw_uid_list,
                             ['76', '768', '642', '930', '329', '633', '716', '871', '543', '754'])

        self.assertSequenceEqual(train_set.iid_list, range(10))
        self.assertListEqual(train_set.raw_iid_list,
                             ['93', '257', '795', '709', '705', '226', '478', '195', '737', '282'])

        train_set = MatrixTrainSet.from_uir(self.triplet_data,
                                            global_uid_map=OrderedDict(),
                                            global_iid_map=OrderedDict(),
                                            global_ui_set=set([('76', '93')]),
                                            verbose=True)

        self.assertEqual(train_set.num_users, 9)
        self.assertEqual(train_set.num_items, 9)

    def test_uir_iter(self):
        train_set = MatrixTrainSet.from_uir(self.triplet_data, global_uid_map={}, global_iid_map={},
                                            global_ui_set=set(), verbose=True)

        users = [batch_users for batch_users, _, _ in train_set.uir_iter()]
        self.assertSequenceEqual(users, range(10))

        items = [batch_items for _, batch_items, _ in train_set.uir_iter()]
        self.assertSequenceEqual(items, range(10))

        ratings = [batch_ratings for _, _, batch_ratings in train_set.uir_iter()]
        self.assertListEqual(ratings, [4, 4, 4, 4, 3, 4, 4, 5, 3, 4])

    def test_uij_iter(self):
        train_set = MatrixTrainSet.from_uir(self.triplet_data, global_uid_map={}, global_iid_map={},
                                            global_ui_set=set(), verbose=True)

        users = [batch_users for batch_users, _, _ in train_set.uij_iter()]
        self.assertSequenceEqual(users, range(10))

        pos_items = [batch_pos_items for _, batch_pos_items, _ in train_set.uij_iter()]
        self.assertSequenceEqual(pos_items, range(10))

        neg_items = [batch_neg_items for _, _, batch_neg_items in train_set.uij_iter()]
        self.assertRaises(AssertionError, self.assertSequenceEqual, neg_items, range(10))

    def test_user_iter(self):
        train_set = MatrixTrainSet.from_uir(self.triplet_data, global_uid_map={}, global_iid_map={},
                                            global_ui_set=set(), verbose=True)

        npt.assert_array_equal(np.arange(10).reshape(10, 1),
                               [u for u in train_set.user_iter()])
        self.assertRaises(AssertionError, npt.assert_array_equal,
                          np.arange(10).reshape(10, 1),
                          [u for u in train_set.user_iter(shuffle=True)])

    def test_item_iter(self):
        train_set = MatrixTrainSet.from_uir(self.triplet_data, global_uid_map={}, global_iid_map={},
                                            global_ui_set=set(), verbose=True)

        npt.assert_array_equal(np.arange(10).reshape(10, 1),
                               [i for i in train_set.item_iter()])
        self.assertRaises(AssertionError, npt.assert_array_equal,
                          np.arange(10).reshape(10, 1),
                          [i for i in train_set.item_iter(shuffle=True)])

    def test_uir_tuple(self):
        train_set = MatrixTrainSet.from_uir(self.triplet_data,
                                            global_uid_map=None,
                                            global_iid_map=None,
                                            global_ui_set=None,
                                            verbose=True)

        self.assertEqual(len(train_set.uir_tuple), 3)
        self.assertEqual(len(train_set.uir_tuple[0]), 10)

        train_set.uir_tuple = None
        self.assertEqual(len(train_set.uir_tuple[1]), 10)
        self.assertEqual(len(train_set.uir_tuple[2]), 10)

        try:
            train_set.uir_tuple = ([], [])
        except ValueError:
            assert True

        self.assertEqual(train_set.num_batches(batch_size=5), 2)


if __name__ == '__main__':
    unittest.main()
