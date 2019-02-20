# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.data import reader
from cornac.data import TestSet


def test_testset():
    """Test TestSet"""
    triplet_data = reader.read_uir('./tests/data.txt')
    test_set = TestSet.from_uir(triplet_data, global_uid_map={}, global_iid_map={}, global_ui_set=set())

    assert test_set.get_uid('768') == 1
    assert test_set.get_iid('195') == 7

    assert all([a == b for a, b in zip(test_set.get_users(), range(10))])
    assert all([a == b for a, b in zip(test_set.get_ratings(2), [(2, 4)])])

    test_set = TestSet.from_uir(triplet_data, global_uid_map={}, global_iid_map={},
                                global_ui_set=set([('76', '93')]), verbose=True)
    assert len(test_set.get_users()) == 9
