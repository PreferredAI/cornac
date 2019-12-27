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
import random
import time

from cornac.datasets import epinions


class TestEpinions(unittest.TestCase):

    def test_load_feedback(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            self.assertEqual(len(epinions.load_feedback()), 664824)

    def test_load_trust(self):
        # only run data download tests 20% of the time to speed up frequent testing
        random.seed(time.time())
        if random.random() > 0.8:
            self.assertEqual(len(epinions.load_trust()), 487183)


if __name__ == '__main__':
    unittest.main()
