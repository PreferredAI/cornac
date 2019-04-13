# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
import numpy.testing as npt
from cornac.utils.init_utils import *


class TestInitUtils(unittest.TestCase):

    def setUp(self):
        self.shape = (2, 3)

    def test_zeros(self):
        npt.assert_array_equal(zeros(self.shape), np.zeros(self.shape, dtype=np.float32))

    def test_ones(self):
        npt.assert_array_equal(ones(self.shape), np.ones(self.shape, dtype=np.float32))

    def test_constant(self):
        npt.assert_array_equal(constant(self.shape, val=2.5), np.ones(self.shape, dtype=np.float32) * 2.5)

    def test_xavier(self):
        std = np.sqrt(2.0 / np.sum(self.shape))
        limit = np.sqrt(3.0) * std
        self.assertTrue(np.min(xavier_uniform(self.shape)) >= -limit)
        self.assertTrue(np.max(xavier_uniform(self.shape)) <= limit)

        seed = np.random.randint(123)
        x_norm = xavier_normal(self.shape, random_state=seed)
        np.random.seed(seed)
        npt.assert_array_equal(x_norm,
                               np.random.normal(0, np.sqrt(2.0 / np.sum(self.shape)), self.shape).astype(np.float32))


if __name__ == '__main__':
    unittest.main()
