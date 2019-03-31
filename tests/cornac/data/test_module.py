# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
import numpy as np
from collections import OrderedDict
from cornac.data import FeatureModule


class TestFeatureModule(unittest.TestCase):

    def setUp(self):
        self.id_feature = {'a': np.zeros(10),
                           'b': np.ones(10)}

    def test_init(self):
        md = FeatureModule()
        md.build(id_map=None)
        self.assertIsNone(md.features)

        md = FeatureModule(features=np.asarray(list(self.id_feature.values())),
                           ids=self.id_feature.keys(),
                           normalized=True)

        global_iid_map = OrderedDict()
        global_iid_map.setdefault('a', len(global_iid_map))
        md.build(id_map=global_iid_map)

        self.assertEqual(md.features.shape[0], 2)
        self.assertEqual(md.features.shape[1], 10)
        self.assertEqual(md.feature_dim, 10)

    def test_batch_feature(self):
        md = FeatureModule(features=np.asarray(list(self.id_feature.values())),
                           ids=self.id_feature.keys(),
                           normalized=True)

        global_iid_map = OrderedDict({'a': 0, 'b': 1})
        md.build(id_map=global_iid_map)

        b = md.batch_feature(batch_ids=[0])
        self.assertEqual(b.shape[0], 1)
        self.assertEqual(b.shape[1], 10)


if __name__ == '__main__':
    unittest.main()
