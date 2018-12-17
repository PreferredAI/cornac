# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""


from ..reader import txt_to_triplets
from ..trainset import MatrixTrainSet

data_file = './cornac/data/tests/triplet_data.txt'
u_col = 0
i_col = 1
r_col = 2
sep = '\t'


def test_MatrixTrainSet():
    """Test MatrixTrainSet"""

    triplet_data = txt_to_triplets(data_file, u_col, i_col, r_col, sep, skip_lines=0)
    train_set = MatrixTrainSet.from_triplets(triplet_data, pre_uid_map={}, pre_iid_map={}, pre_ur_set=set())

    assert train_set.num_users == 10
    assert train_set.num_items == 10

    assert train_set.is_unk_user(7) == False
    assert train_set.is_unk_user(13) == True

    assert train_set.is_unk_item(3) == False
    assert train_set.is_unk_item(16) == True

    assert train_set.min_rating == 3
    assert train_set.max_rating == 5

    assert train_set.get_uid('768') == 1
    assert train_set.get_iid('195') == 7

    assert all([a == b for a, b in zip(train_set.get_uid_list(), range(10))])
    assert all([a == b for a, b in zip(train_set.get_raw_uid_list(),
                                       ['76', '768', '642', '930', '329', '633', '716', '871', '543', '754'])])

    assert all([a == b for a, b in zip(train_set.get_iid_list(), range(10))])
    assert all([a == b for a, b in zip(train_set.get_raw_iid_list(),
                                       ['93', '257', '795', '709', '705', '226', '478', '195', '737', '282'])])
