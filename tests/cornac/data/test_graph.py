# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import unittest
from cornac.data import GraphModule
from cornac.data import Reader
from collections import OrderedDict


class TestGraphModule(unittest.TestCase):

    def test_init(self):
        data = Reader().read('./tests/graph_data.txt', sep=' ')
        gmd = GraphModule(data=data)

        self.assertEqual(len(gmd.raw_data), 7)
        self.assertEqual(len(gmd.map_rid), 0)
        self.assertEqual(len(gmd.map_cid), 0)
        self.assertEqual(len(gmd.val), 0)
        self.assertIsNone(gmd.matrix)

    def test_build(self):
        data = Reader().read('./tests/graph_data.txt', sep=' ')
        gmd = GraphModule(data=data)

        global_iid_map = OrderedDict()
        for raw_iid, raw_jid, val in data:
            mapped_iid = global_iid_map.setdefault(raw_iid, len(global_iid_map))

        gmd.build(id_map=global_iid_map)

        self.assertEqual(len(gmd.map_rid), 7)
        self.assertEqual(len(gmd.map_cid), 7)
        self.assertEqual(len(gmd.val), 7)
        self.assertEqual(gmd.matrix.shape, (7, 3))




if __name__ == '__main__':
    unittest.main()
