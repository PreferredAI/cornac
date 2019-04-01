# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
import random
import time
from cornac.datasets import citeulike


class TestCiteULike(unittest.TestCase):

    def test_load_data(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            data = citeulike.load_data()
            self.assertEqual(len(data), 210537)


if __name__ == '__main__':
    unittest.main()
