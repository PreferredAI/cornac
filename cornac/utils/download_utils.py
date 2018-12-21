# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import os
import zipfile
from urllib import request


def urlretrieve(url, fpath):
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}

    req = request.Request(url, headers=headers)

    response = request.urlopen(req)
    with open(fpath, 'wb') as f:
        f.write(response.read())


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

        print('Downloading data from', self.url)
        print('and save to', fpath)

        if self.unzip:
            tmp_path = os.path.join(download_dir, 'tmp.zip')
            urlretrieve(self.url, tmp_path)

            if verbose:
                print('Unziping...')
            with zipfile.ZipFile(tmp_path, 'r') as tmp_zip:
                tmp_zip.extractall(download_dir)
            os.remove(tmp_path)
        else:
            urlretrieve(self.url, fpath)

        if verbose:
            print('Downloading finished!')

        return fpath
