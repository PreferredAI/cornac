# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import unittest

import numpy as np
import numpy.testing as npt
import scipy.sparse as sp

from cornac.utils.common import sigmoid
from cornac.utils.common import safe_indexing
from cornac.utils.common import validate_format
from cornac.utils.common import scale
from cornac.utils.common import clip
from cornac.utils.common import excepts
from cornac.utils.common import intersects
from cornac.utils.common import estimate_batches
from cornac.utils.common import get_rng
from cornac.utils.common import normalize


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
        self.assertEqual(1, scale(5, 0, 1, 5, 5))

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

    def test_get_rng(self):
        try:
            get_rng('a')
        except ValueError:
            assert True

    def test_normalize(self):
        """
        X = array([[1., 0., 2.],
                   [0., 0., 3.],
                   [4., 5., 6.]])
        """
        indptr = np.array([0, 2, 3, 6])
        indices = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1., 2., 3., 4., 5., 6.], dtype=np.float64)
        X = sp.csr_matrix((data, indices, indptr), shape=(3, 3))
        XA = X.toarray()

        # normalizing rows (axis=1)
        X_l1 = XA / (np.abs(XA).sum(1).reshape(-1, 1))
        X_l2 = XA / (np.sqrt((XA ** 2).sum(1)).reshape(-1, 1))
        X_max = XA / (np.max(XA, axis=1).reshape(-1, 1))
        # sparse input
        npt.assert_array_equal(X_l1, normalize(X, "l1", axis=1, copy=True).toarray())
        npt.assert_array_equal(X_l2, normalize(X, "l2", axis=1, copy=True).toarray())
        npt.assert_array_equal(X_max, normalize(X, "max", axis=1, copy=True).toarray())
        # dense input
        npt.assert_array_equal(X_l1, normalize(XA, "l1", axis=1, copy=True))
        npt.assert_array_equal(X_l2, normalize(XA, "l2", axis=1, copy=True))
        npt.assert_array_equal(X_max, normalize(XA, "max", axis=1, copy=True))

        # normalizing columns (axis=0)
        X_l1 = XA / (np.abs(XA).sum(0).reshape(1, -1))
        X_l2 = XA / (np.sqrt((XA ** 2).sum(0)).reshape(1, -1))
        X_max = XA / (np.max(XA, axis=0).reshape(1, -1))
        # sparse input
        npt.assert_array_equal(X_l1, normalize(X, "l1", axis=0, copy=True).toarray())
        npt.assert_array_equal(X_l2, normalize(X, "l2", axis=0, copy=True).toarray())
        npt.assert_array_equal(X_max, normalize(X, "max", axis=0, copy=True).toarray())
        # dense input
        npt.assert_array_equal(X_l1, normalize(XA, "l1", axis=0, copy=True))
        npt.assert_array_equal(X_l2, normalize(XA, "l2", axis=0, copy=True))
        npt.assert_array_equal(X_max, normalize(XA, "max", axis=0, copy=True))

        # check valid norm type
        try:
            normalize(X, norm='bla bla')
        except ValueError:
            assert True

        # check valid input shape
        try:
            normalize(XA[:, np.newaxis])
        except ValueError:
            assert True

        # copy=True, sparse
        normalized_X = normalize(X, copy=True)
        self.assertFalse(np.allclose(X.data, normalized_X.data))

        # copy=True, dense
        normalized_XA = normalize(XA, copy=True)
        self.assertFalse(np.allclose(XA, normalized_XA))

        # copy=False, sparse
        original = X.data.copy()
        normalized_X = normalize(X, copy=False)
        self.assertFalse(np.allclose(original, X.data))
        npt.assert_array_equal(normalized_X.data, X.data)

        # copy=False, dense
        original = XA.copy()
        normalized_XA = normalize(XA, copy=False)
        self.assertFalse(np.allclose(original, XA))
        npt.assert_array_equal(normalized_XA, XA)


if __name__ == '__main__':
    unittest.main()
