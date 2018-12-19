# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import os
import zipfile
from six.moves.urllib.request import urlretrieve



class DownloadItem:

    def __init__(self, url, relative_path, unzip=False, cache_dir=None, sub_dir='datasets'):
        self.url = url
        self.rel_path = relative_path
        self.unzip = unzip
        self.cache_dir = cache_dir
        self.sub_dir = sub_dir


    def _get_download_dir(self):
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.path.expanduser('~'), '.cornac')

        base_dir = os.path.expanduser(self.cache_dir)
        if not os.access(base_dir, os.W_OK):
            base_dir = os.path.join('/tmp', '.cornac')

        download_dir = os.path.join(base_dir, self.sub_dir)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        return download_dir


    def download_if_needed(self, verbose=False):
        download_dir = self._get_download_dir()
        fpath = os.path.join(download_dir, self.rel_path)

        if os.path.exists(fpath):
            return fpath

        if verbose:
            print('Downloading data from', self.url)
            print('and save to', fpath)

        if self.unzip:
            tmp_path = os.path.join(download_dir, 'tmp.zip')
            urlretrieve(self.url, tmp_path)

            with zipfile.ZipFile(tmp_path, 'r') as tmp_zip:
                tmp_zip.extractall(download_dir)
            os.remove(tmp_path)
        else:
            urlretrieve(self.url, fpath)

        return fpath