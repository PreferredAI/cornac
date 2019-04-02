# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
from cornac.data import Reader
from cornac.data import TestSet
from collections import OrderedDict


class TestTestSet(unittest.TestCase):

    def test_init(self):
        triplet_data = Reader().read('./tests/data.txt')
        test_set = TestSet.from_uir(triplet_data, global_uid_map={}, global_iid_map={}, global_ui_set=set())

        self.assertEqual(test_set.get_uid('768'), 1)
        self.assertEqual(test_set.get_iid('195'), 7)

        self.assertSequenceEqual(test_set.users, range(10))
        self.assertListEqual(test_set.get_ratings(2), [(2, 4)])

        test_set = TestSet.from_uir(triplet_data,
                                    global_uid_map=OrderedDict(),
                                    global_iid_map=OrderedDict(),
                                    global_ui_set=set([('76', '93')]),
                                    verbose=True)
        self.assertEqual(len(test_set.users), 9)


if __name__ == '__main__':
    unittest.main()
