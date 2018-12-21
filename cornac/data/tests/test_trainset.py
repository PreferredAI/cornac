# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""



def test_trainset():
    """Test TrainSet"""
    
    from collections import OrderedDict

    uid_map = OrderedDict([('a', 0), ('b', 1)])
    iid_map = OrderedDict([('x', 0), ('y', 1), ('z', 2)])

    from ..trainset import TrainSet
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


def test_matrix_trainset():
    """Test MatrixTrainSet"""

    data_file = './cornac/data/tests/data.txt'
    u_col = 0
    i_col = 1
    r_col = 2
    sep = '\t'

    from ..reader import txt_to_uir_triplets
    triplet_data = txt_to_uir_triplets(data_file, u_col, i_col, r_col, sep, skip_lines=0)

    from ..trainset import MatrixTrainSet
    train_set = MatrixTrainSet.from_uir_triplets(triplet_data, pre_uid_map={}, pre_iid_map={}, pre_ui_set=set(), verbose=True)

    assert train_set.matrix.shape == (10, 10)
    assert train_set.min_rating == 3
    assert train_set.max_rating == 5

    assert int(train_set.global_mean) == int((3*2 + 4*7 + 5) / 10)

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


    train_set = MatrixTrainSet.from_uir_triplets(triplet_data, pre_uid_map={}, pre_iid_map={},
                                                 pre_ui_set=set([('76', '93')]), verbose=True)
    assert train_set.num_users == 9
    assert train_set.num_items == 9