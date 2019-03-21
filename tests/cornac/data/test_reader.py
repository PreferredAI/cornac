# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
from cornac.data import reader


class TestReader(unittest.TestCase):

    def test_read_ui(self):
        data_file = './tests/data.txt'
        triplets = reader.read_ui(data_file, value=2.0)

        self.assertEqual(len(triplets), 30)
        self.assertEqual(triplets[0][1], '93')
        self.assertEqual(triplets[1][2], 2.0)

    def test_read_uir(self):
        data_file = './tests/data.txt'
        triplet_data = reader.read_uir(data_file)

        self.assertEqual(len(triplet_data), 10)
        self.assertEqual(triplet_data[4][2], 3)
        self.assertEqual(triplet_data[6][1], '478')
        self.assertEqual(triplet_data[8][0], '543')

        try:
            reader.read_uir(data_file, 10)
        except IndexError:
            assert True


if __name__ == '__main__':
    unittest.main()
