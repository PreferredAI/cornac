# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
from cornac.utils import tryimport


class TestDummyImport(unittest.TestCase):

    def test_tryimport(self):
        dummy = tryimport('this_module_could_not_exist_bla_bla')
        try:
            dummy.some_attribute
        except ImportError:
            assert True


if __name__ == '__main__':
    unittest.main()
