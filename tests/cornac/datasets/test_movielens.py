# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
import random
import time
from cornac.datasets import movielens


class TestMovieLens(unittest.TestCase):

    def test_load_100k(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            ml_100k = movielens.load_100k()
            self.assertEqual(len(ml_100k), 100000)

    def test_load_1m(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            ml_1m = movielens.load_1m()
            self.assertEqual(len(ml_1m), 1000209)

    def test_load_plot(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            plots, ids = movielens.load_plot()
            self.assertEqual(len(ids), 10076)


if __name__ == '__main__':
    unittest.main()
