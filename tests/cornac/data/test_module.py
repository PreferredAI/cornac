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
        self.id_feature = {'a': np.zeros(10)}

    def test_init(self):
        md = FeatureModule()
        md.build(global_id_map=None)
        self.assertIsNone(md.features)

        md = FeatureModule(id_feature=self.id_feature, normalized=True)

        global_iid_map = OrderedDict()
        global_iid_map.setdefault('a', len(global_iid_map))
        md.build(global_id_map=global_iid_map)

        self.assertEqual(md.features.shape[0], 1)
        self.assertEqual(md.features.shape[1], 10)
        self.assertEqual(md.feat_dim, 10)
        self.assertEqual(len(md._id_feature), 0)

    def test_batch_feat(self):
        md = FeatureModule(id_feature=self.id_feature, normalized=True)

        global_iid_map = OrderedDict({'a': 0})
        md.build(global_id_map=global_iid_map)

        b = md.batch_feat(batch_ids=[0])
        self.assertEqual(b.shape[0], 1)
        self.assertEqual(b.shape[1], 10)


if __name__ == '__main__':
    unittest.main()
