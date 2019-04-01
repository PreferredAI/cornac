# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
from cornac.data import ImageModule


class TestImageModule(unittest.TestCase):

    def test_init(self):
        md = ImageModule()
        md.build(id_map=None)

        self.assertIsNone(md.images)
        self.assertIsNone(md.paths)

    def batch_image(self):
        md = ImageModule()
        md.build(id_map=None)

        try:
            md.batch_image(batch_ids=None)
        except:
            raise NotImplementedError


if __name__ == '__main__':
    unittest.main()
