# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
from cornac.data import ImageModule


class TestImageModule(unittest.TestCase):

    def test_init(self):
        md = ImageModule()
        md.build(global_id_map=None)

        self.assertIsNone(md.data_image)
        self.assertIsNone(md.data_path)

    def batch_image(self):
        md = ImageModule()
        md.build(global_id_map=None)
        md.batch_image(batch_ids=None)


if __name__ == '__main__':
    unittest.main()