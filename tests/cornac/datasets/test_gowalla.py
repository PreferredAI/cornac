# Copyright 2023 The Cornac Authors. All Rights Reserved.
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

import random
import time
import unittest

from cornac.datasets import gowalla


class TestGowalla(unittest.TestCase):

    def test_load_checkins(self):
        random.seed(time.time())
        if random.random() > 0.8:
            checkins = gowalla.load_checkins()
            self.assertEqual(len(checkins), 6442892)


if __name__ == "__main__":
    unittest.main()
