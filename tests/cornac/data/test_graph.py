# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import unittest
from cornac.data import GraphModule
from cornac.data import reader


class TestGraphModule(unittest.TestCase):

    def load_data(self):
        self.data = reader.read_uir('./tests/data.txt')


if __name__ == '__main__':
    unittest.main()

