# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""



data_file = './cornac/data/tests/data.txt'
u_col = 0
i_col = 1
r_col = 2
sep = '\t'


def test_testset():
    """Test TestSet"""

    from ..reader import txt_to_uir_triplets
    triplet_data = txt_to_uir_triplets(data_file, u_col, i_col, r_col, sep, skip_lines=0)

    from ..testset import TestSet
    test_set = TestSet.from_uir_triplets(triplet_data, pre_uid_map={}, pre_iid_map={}, pre_ui_set=set())

    assert test_set.get_uid('768') == 1
    assert test_set.get_iid('195') == 7

    assert all([a == b for a, b in zip(test_set.get_users(), range(10))])

    assert all([a == b for a, b in zip(test_set.get_ratings(2), [(2, 4)])])

    test_set = TestSet.from_uir_triplets(triplet_data, pre_uid_map={}, pre_iid_map={},
                                         pre_ui_set=set([('76', '93')]), verbose=True)
    assert len(test_set.get_users()) == 9


