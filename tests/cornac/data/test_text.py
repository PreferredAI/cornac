# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.data import TextModule

def test_init():
    md = TextModule()
    md.build(global_id_map=None)
