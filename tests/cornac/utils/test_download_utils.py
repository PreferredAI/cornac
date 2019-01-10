# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import os
from cornac.utils.download_utils import DownloadItem


def test_download_normal_file():
    download_item = DownloadItem(url='https://static.preferred.ai/cornac/hello_world.txt',
                                 relative_path='hello_word.txt')
    fpath = download_item.maybe_download(verbose=True)

    assert os.path.exists(fpath)

    with open(fpath, 'r') as f:
        assert "I'm Cornac!" == f.read().strip()


def test_download_zip_file():
    download_item = DownloadItem(url='https://static.preferred.ai/cornac/dummy.zip',
                                 relative_path='dummy/hello_world.txt', unzip=True)
    fpath = download_item.maybe_download(verbose=True)

    assert os.path.exists(fpath)

    with open(fpath, 'r') as f:
        assert "I'm Cornac!" == f.read().strip()
