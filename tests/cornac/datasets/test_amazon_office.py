# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import unittest
import random
import time
from cornac.datasets import amazon_office as office


class TestAmazonOffice(unittest.TestCase):

    def test_amazon_office(self):
        random.seed(time.time())
        if random.random() > 0.8:
            ratings = office.load_rating()
            contexts = office.load_context()
            self.assertEqual(len(ratings), 53282)
            self.assertEqual(len(contexts), 108466)


if __name__ == '__main__':
    unittest.main()
