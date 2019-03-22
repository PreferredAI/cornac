# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import unittest
from cornac.data import GraphModule
from cornac.data import reader


class TestGraphModule(unittest.TestCase):

    def load_data(self):
        self.data = reader.read_uir('./tests/graph_data.txt')

    def test_init(self):
        gmd = GraphModule(data=data)
        gmd.build(global_id_map=None)

if __name__ == '__main__':
    unittest.main()

