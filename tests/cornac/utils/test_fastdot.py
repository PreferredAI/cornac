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
