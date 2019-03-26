# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
import numpy as np
import numpy.testing as npt
from cornac.utils.common import sigmoid
from cornac.utils.common import safe_indexing
from cornac.utils.common import validate_format
from cornac.utils.common import scale
from cornac.utils.common import clip
from cornac.utils.common import excepts
from cornac.utils.common import intersects
from cornac.utils.common import estimate_batches


class TestCommon(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(0, sigmoid(-np.inf))
        self.assertEqual(0.5, sigmoid(0))
        self.assertEqual(1, sigmoid(np.inf))

        self.assertGreater(0.5, sigmoid(-0.1))
        self.assertGreater(sigmoid(0.1), 0.5)

    def test_scale(self):
        self.assertEqual(1, scale(0, 1, 5, 0, 1))
        self.assertEqual(3, scale(0.5, 1, 5, 0, 1))
        self.assertEqual(5, scale(1, 1, 5, 0, 1))

        npt.assert_array_equal(scale(np.asarray([0, 0.25, 0.5, 0.75, 1]), 1, 5),
                               np.asarray([1, 2, 3, 4, 5]))

    def test_clip(self):
        self.assertEqual(1, clip(0, 1, 5))
        self.assertEqual(3, clip(3, 1, 5))
        self.assertEqual(5, clip(6, 1, 5))

        npt.assert_array_equal(clip(np.asarray([0, 3, 6]), 1, 5),
                               np.asarray([1, 3, 5]))

    def test_intersects(self):
        self.assertEqual(0, len(intersects(np.asarray([1]), np.asarray(2))))
        self.assertEqual(1, len(intersects(np.asarray([2]), np.asarray(2))))

        npt.assert_array_equal(intersects(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                               np.asarray([2, 1]))

    def test_excepts(self):
        self.assertEqual(1, len(excepts(np.asarray([1]), np.asarray(2))))
        self.assertEqual(0, len(excepts(np.asarray([2]), np.asarray(2))))

        npt.assert_array_equal(excepts(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                               np.asarray([3]))

    def test_safe_indexing(self):
        npt.assert_array_equal(safe_indexing(np.asarray([3, 2, 1]), np.asarray([1, 2])),
                               np.asarray([2, 1]))
        npt.assert_array_equal(safe_indexing(np.asarray([3, 2, 1]), [1, 2]),
                               np.asarray([2, 1]))
        self.assertListEqual(safe_indexing([3, 2, 1], [1, 2]), [2, 1])

    def test_validate_format(self):
        self.assertEqual('UIR', validate_format('UIR', ['UIR']))
        self.assertEqual('UIRT', validate_format('UIRT', ['UIRT']))

        try:
            validate_format('iur', ['UIR'])
        except ValueError:
            assert True

    def test_estimate_batches(self):
        self.assertEqual(estimate_batches(3, 2), 2)
        self.assertEqual(estimate_batches(4, 2), 2)
        self.assertEqual(estimate_batches(1, 2), 1)


if __name__ == '__main__':
    unittest.main()
