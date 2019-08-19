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

from collections import OrderedDict
from cornac.data import FeatureModality


class TestFeatureModality(unittest.TestCase):

    def setUp(self):
        self.id_feature = {'a': np.zeros(10),
                           'b': np.ones(10)}

    def test_init(self):
        md = FeatureModality()
        md.build(id_map=None)
        self.assertIsNone(md.features)

        md = FeatureModality(features=np.asarray(list(self.id_feature.values())),
                           ids=list(self.id_feature.keys()),
                           normalized=True)

        global_iid_map = OrderedDict()
        global_iid_map.setdefault('a', len(global_iid_map))
        md.build(id_map=global_iid_map)

        self.assertEqual(md.features.shape[0], 2)
        self.assertEqual(md.features.shape[1], 10)
        self.assertEqual(md.feature_dim, 10)

    def test_batch_feature(self):
        md = FeatureModality(features=np.asarray(list(self.id_feature.values())),
                           ids=list(self.id_feature.keys()),
                           normalized=True)

        global_iid_map = OrderedDict({'a': 0, 'b': 1})
        md.build(id_map=global_iid_map)

        b = md.batch_feature(batch_ids=[0])
        self.assertEqual(b.shape[0], 1)
        self.assertEqual(b.shape[1], 10)


if __name__ == '__main__':
    unittest.main()
