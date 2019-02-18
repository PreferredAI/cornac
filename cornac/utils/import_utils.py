# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from types import ModuleType

class DummyModule(ModuleType):
    def __getattr__(self, key):
        return None
    __all__ = []   # support wildcard imports

def tryimport(name, globals=None, locals=None, fromlist=[], level=0):
    try:
        return __import__(name, globals, locals, fromlist, level)
    except ImportError:
        return DummyModule(name)