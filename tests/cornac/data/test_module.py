# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.data import Module
import numpy as np
from collections import OrderedDict


def test_init():
    md = Module()
    md.build(ordered_ids=None)
    assert md.data_feature is None

    id_feature = {'a': np.zeros(10)}
    md = Module(id_feature=id_feature, normalized=True)

    global_iid_map = OrderedDict()
    global_iid_map.setdefault('a', len(global_iid_map))
    md.build(ordered_ids=global_iid_map)

    assert md.data_feature.shape[0] == 1
    assert md.data_feature.shape[1] == 10
    assert md.feature_dim == 10
    assert len(md._id_feature) == 0


def test_batch_feature():
    id_feature = {'a': np.zeros(10)}
    md = Module(id_feature=id_feature, normalized=True)

    global_iid_map = OrderedDict()
    global_iid_map.setdefault('a', len(global_iid_map))
    md.build(ordered_ids=global_iid_map)

    b = md.batch_feature(batch_ids=[0])
    assert b.shape[0] == 1
    assert b.shape[1] == 10