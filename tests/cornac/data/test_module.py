# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.data import FeatureModule
import numpy as np
from collections import OrderedDict


def test_init():
    md = FeatureModule()
    md.build(global_id_map=None)
    assert md.features is None

    id_feature = {'a': np.zeros(10)}
    md = FeatureModule(id_feature=id_feature, normalized=True)

    global_iid_map = OrderedDict()
    global_iid_map.setdefault('a', len(global_iid_map))
    md.build(global_id_map=global_iid_map)

    assert md.features.shape[0] == 1
    assert md.features.shape[1] == 10
    assert md.feat_dim == 10
    assert len(md._id_feature) == 0


def test_batch_feature():
    id_feature = {'a': np.zeros(10)}
    md = FeatureModule(id_feature=id_feature, normalized=True)

    global_iid_map = OrderedDict()
    global_iid_map.setdefault('a', len(global_iid_map))
    md.build(global_id_map=global_iid_map)

    b = md.batch_feat(batch_ids=[0])
    assert b.shape[0] == 1
    assert b.shape[1] == 10