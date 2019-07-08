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

import os
import unittest

from cornac.utils.download import cache


class TestDownload(unittest.TestCase):

    def test_download_normal_file(self):
        fpath = cache(url='https://static.preferred.ai/cornac/tests/hello_world.txt')
        self.assertTrue(os.path.exists(fpath))
        with open(fpath, 'r') as f:
            self.assertEqual("I'm Cornac!", f.read().strip())

        cache(url='https://static.preferred.ai/cornac/tests/hello_world.txt')

    def test_download_zip_file(self):
        fpath = cache(url='https://static.preferred.ai/cornac/tests/dummy.zip',
                      unzip=True, relative_path='dummy/hello_world.txt')
        self.assertTrue(os.path.exists(fpath))
        with open(fpath, 'r') as f:
            self.assertEqual("I'm Cornac!", f.read().strip())

    def test_download_gzip_file(self):
        fpath = cache(url='https://static.preferred.ai/cornac/tests/dummy.tar.gz',
                      unzip=True, relative_path='dummy/gz.txt')
        self.assertTrue(os.path.exists(fpath))
        with open(fpath, 'r') as f:
            self.assertEqual('gz', f.read().strip())

    def test_download_bzip2_file(self):
        fpath = cache(url='https://static.preferred.ai/cornac/tests/dummy.tar.bz2',
                      unzip=True, relative_path='dummy/bz2.txt')
        self.assertTrue(os.path.exists(fpath))
        with open(fpath, 'r') as f:
            self.assertEqual('bz2', f.read().strip())


if __name__ == '__main__':
    unittest.main()
