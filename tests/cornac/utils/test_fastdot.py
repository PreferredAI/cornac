# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
import numpy as np
import numpy.testing as npt
from cornac.utils import fast_dot


class TestFastDot(unittest.TestCase):

    def test_fast_dot(self):
        vec = np.ones(2, dtype=np.float32)
        mat = np.ones((2, 2), dtype=np.float32)
        output = np.zeros(mat.shape[0], dtype=np.float32)
        fast_dot(vec, mat, output)
        npt.assert_array_equal(np.asarray([2, 2]), output)

        vec = np.asarray([1, 2], dtype=np.double)
        mat = np.asarray([[1, 2], [3, 4]], dtype=np.double)
        output = np.zeros(mat.shape[0], dtype=np.double)
        fast_dot(vec, mat, output)
        npt.assert_array_equal(np.asarray([5, 11]), output)


if __name__ == '__main__':
    unittest.main()
