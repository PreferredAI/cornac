# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from . import FeatureModule


class TextModule(FeatureModule):
    """Text module

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, global_id_map):
        pass