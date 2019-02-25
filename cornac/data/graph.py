# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from . import Module

class GraphModule(Module):
    """Graph module

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, ordered_ids):
        pass