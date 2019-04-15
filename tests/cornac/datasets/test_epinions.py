# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
import random
import time
from cornac.datasets import epinions


class TestEpinions(unittest.TestCase):

    def test_load_data(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            self.assertEqual(len(epinions.load_data()), 664824)

    def test_load_trust(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            self.assertEqual(len(epinions.load_trust()), 487183)


if __name__ == '__main__':
    unittest.main()
