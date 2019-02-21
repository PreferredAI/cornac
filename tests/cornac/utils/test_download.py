# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import os
from cornac.utils.download import cache


def test_download_normal_file():
    fpath = cache(url='https://static.preferred.ai/cornac/hello_world.txt',
                  relative_path='hello_world.txt')
    assert os.path.exists(fpath)
    with open(fpath, 'r') as f:
        assert "I'm Cornac!" == f.read().strip()


def test_download_zip_file():
    fpath = cache(url='https://static.preferred.ai/cornac/dummy.zip',
                  unzip=True, relative_path='dummy/hello_world.txt')
    assert os.path.exists(fpath)
    with open(fpath, 'r') as f:
        assert "I'm Cornac!" == f.read().strip()
