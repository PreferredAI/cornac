# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.data import ImageModule

def test_init():
    md = ImageModule()
    md.build(global_id_map=None)

    assert md.data_image is None
    assert md.data_path is None


def batch_image():
    md = ImageModule()
    md.build(global_id_map=None)

    md.batch_image(batch_ids=None)