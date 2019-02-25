# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.data import reader
from cornac.data import TrainSet
from cornac.data import MatrixTrainSet

from collections import OrderedDict
import numpy as np


def test_trainset():
    """Test TrainSet"""
    uid_map = OrderedDict([('a', 0), ('b', 1)])
    iid_map = OrderedDict([('x', 0), ('y', 1), ('z', 2)])

    train_set = TrainSet(uid_map, iid_map)

    assert train_set.num_users == 2
    assert train_set.num_items == 3

    assert train_set.is_unk_user(1) == False
    assert train_set.is_unk_user(2) == True

    assert train_set.is_unk_item(2) == False
    assert train_set.is_unk_item(4) == True

    assert train_set.get_uid('b') == 1
    assert train_set.get_iid('y') == 1

    assert all([a == b for a, b in zip(train_set.get_uid_list(), [0, 1])])
    assert all([a == b for a, b in zip(train_set.get_raw_uid_list(), ['a', 'b'])])

    assert all([a == b for a, b in zip(train_set.get_iid_list(), [0, 1, 2])])
    assert all([a == b for a, b in zip(train_set.get_raw_iid_list(), ['x', 'y', 'z'])])


def test_trainset_idx_iter():
    ids = [batch_ids for batch_ids in TrainSet.idx_iter(idx_range=10, batch_size=1, shuffle=False)]
    assert all([a == b for a, b in zip(ids, range(10))])

    ids = [batch_ids for batch_ids in TrainSet.idx_iter(idx_range=10, batch_size=1, shuffle=True)]
    assert np.any([a != b for a, b in zip(ids, range(10))])


def test_matrix_trainset():
    """Test MatrixTrainSet"""
    triplet_data = reader.read_uir('./tests/data.txt')
    train_set = MatrixTrainSet.from_uir(triplet_data, global_uid_map={}, global_iid_map={}, global_ui_set=set(),
                                        verbose=True)

    assert train_set.matrix.shape == (10, 10)
    assert train_set.min_rating == 3
    assert train_set.max_rating == 5

    assert int(train_set.global_mean) == int((3 * 2 + 4 * 7 + 5) / 10)

    assert all([a == b for a, b in zip(train_set.item_ppl_rank,
                                       [7, 9, 6, 5, 3, 2, 1, 0, 8, 4])])

    assert train_set.num_users == 10
    assert train_set.num_items == 10

    assert train_set.is_unk_user(7) == False
    assert train_set.is_unk_user(13) == True

    assert train_set.is_unk_item(3) == False
    assert train_set.is_unk_item(16) == True

    assert train_set.get_uid('768') == 1
    assert train_set.get_iid('195') == 7

    assert all([a == b for a, b in zip(train_set.get_uid_list(), range(10))])
    assert all([a == b for a, b in zip(train_set.get_raw_uid_list(),
                                       ['76', '768', '642', '930', '329', '633', '716', '871', '543', '754'])])

    assert all([a == b for a, b in zip(train_set.get_iid_list(), range(10))])
    assert all([a == b for a, b in zip(train_set.get_raw_iid_list(),
                                       ['93', '257', '795', '709', '705', '226', '478', '195', '737', '282'])])

    train_set = MatrixTrainSet.from_uir(triplet_data, global_uid_map={}, global_iid_map={},
                                        global_ui_set=set([('76', '93')]), verbose=True)
    assert train_set.num_users == 9
    assert train_set.num_items == 9


def test_matrix_trainset_uir_iter():
    triplet_data = reader.read_uir('./tests/data.txt')
    train_set = MatrixTrainSet.from_uir(triplet_data, global_uid_map={}, global_iid_map={},
                                        global_ui_set=set(), verbose=True)

    users = [batch_users for batch_users, _, _ in train_set.uir_iter()]
    assert all([a == b for a, b in zip(users, range(10))])

    items = [batch_items for _, batch_items, _ in train_set.uir_iter()]
    assert all([a == b for a, b in zip(items, range(10))])

    ratings = [batch_ratings for _, _, batch_ratings in train_set.uir_iter()]
    assert all([a == b for a, b in zip(ratings, [4, 4, 4, 4, 3, 4, 4, 5, 3, 4])])



def test_matrix_trainset_uij_iter():
    triplet_data = reader.read_uir('./tests/data.txt')
    train_set = MatrixTrainSet.from_uir(triplet_data, global_uid_map={}, global_iid_map={},
                                        global_ui_set=set(), verbose=True)

    users = [batch_users for batch_users, _, _ in train_set.uij_iter()]
    assert all([a == b for a, b in zip(users, range(10))])

    pos_items = [batch_pos_items for _, batch_pos_items, _ in train_set.uij_iter()]
    assert all([a == b for a, b in zip(pos_items, range(10))])

    neg_items = [batch_neg_items for _, _, batch_neg_items in train_set.uij_iter()]
    assert all([a != b for a, b in zip(neg_items, range(10))])


def test_uir_tuple():
    triplet_data = reader.read_uir('./tests/data.txt')
    train_set = MatrixTrainSet.from_uir(triplet_data,
                                        global_uid_map=None,
                                        global_iid_map=None,
                                        global_ui_set=None,
                                        verbose=True)

    try:
        train_set.uir_tuple = ([], [])
    except ValueError:
        assert True

    assert 2 == train_set.num_batches(batch_size=5)