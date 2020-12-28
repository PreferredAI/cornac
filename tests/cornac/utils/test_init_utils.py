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

    def test_gamma(self):
        seed = np.random.randint(123)
        size = 2
        sh = 0.3
        sc= 1.
        x_gamma = gamma(sh,sc,size,seed)
        np.random.seed(seed)
        npt.assert_array_equal(x_gamma,
                               np.random.gamma(sh, sc, size).astype(np.float32))

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
