# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.utils import tryimport

def test_tryimport():
    dummy = tryimport('this_module_could_not_exist_bla_bla')
    try:
        dummy.some_attribute
    except ImportError:
        assert True