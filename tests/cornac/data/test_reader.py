# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
from cornac.data import Reader
from cornac.data.reader import read_text


class TestReader(unittest.TestCase):

    def setUp(self):
        self.data_file = './tests/data.txt'
        self.reader = Reader()

    def test_raise(self):
        try:
            self.reader.read(self.data_file, fmt='bla bla')
        except ValueError:
            assert True

    def test_read_ui(self):
        triplets = self.reader.read(self.data_file, fmt='UI')
        self.assertEqual(len(triplets), 30)
        self.assertEqual(triplets[0][1], '93')
        self.assertEqual(triplets[1][2], 1.0)

        triplets = self.reader.read(self.data_file, fmt='UI', id_inline=True)
        self.assertEqual(len(triplets), 40)

    def test_read_uir(self):
        triplet_data = self.reader.read(self.data_file)

        self.assertEqual(len(triplet_data), 10)
        self.assertEqual(triplet_data[4][2], 3)
        self.assertEqual(triplet_data[6][1], '478')
        self.assertEqual(triplet_data[8][0], '543')

    def test_filter(self):
        reader = Reader(min_user_freq=2)
        self.assertEqual(len(reader.read(self.data_file)), 0)

        reader = Reader(min_item_freq=2)
        self.assertEqual(len(reader.read(self.data_file)), 0)

        reader = Reader(user_set=['76'], item_set=['93'])
        self.assertEqual(len(reader.read(self.data_file)), 1)

        reader = Reader(user_set=['76', '768'])
        self.assertEqual(len(reader.read(self.data_file)), 2)

        reader = Reader(item_set=['93', '257', '795'])
        self.assertEqual(len(reader.read(self.data_file)), 3)

    def test_read_text(self):
        self.assertEqual(len(read_text(self.data_file, sep=None)), 10)
        self.assertEqual(read_text(self.data_file, sep='\t')[1][0], '76')


if __name__ == '__main__':
    unittest.main()
