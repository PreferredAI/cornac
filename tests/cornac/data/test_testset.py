# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.data import Reader
from cornac.data import TestSet


def test_testset():
    """Test TestSet"""
    data_file = './tests/data.txt'
    u_col = 0
    i_col = 1
    r_col = 2
    sep = '\t'

    triplet_data = Reader.read_uir_triplets(data_file, u_col, i_col, r_col, sep, skip_lines=0)

    test_set = TestSet.from_uir_triplets(triplet_data, global_uid_map={}, global_iid_map={}, global_ui_set=set())

    assert test_set.get_uid('768') == 1
    assert test_set.get_iid('195') == 7

    assert all([a == b for a, b in zip(test_set.get_users(), range(10))])

    assert all([a == b for a, b in zip(test_set.get_ratings(2), [(2, 4)])])

    test_set = TestSet.from_uir_triplets(triplet_data, global_uid_map={}, global_iid_map={},
                                         global_ui_set=set([('76', '93')]), verbose=True)
    assert len(test_set.get_users()) == 9
