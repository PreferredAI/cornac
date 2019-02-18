# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import os
import zipfile
from urllib import request


def urlretrieve(url, fpath):
    """Retrieve data from given url

    Parameters
    ----------
    url: str
        The url to the data.

    fpath: str
        The path to file where data is stored.

    """
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
    """Item to be downloaded

    Parameters
    ----------
    url: str
        The url to the data.

    relative_path: str
        Relative path to the data file after finishing the download.

    unzip: bool, optional, default: False
        Whether the data is a zip file and going to be unzipped after the download.

    cache_dir: bool, optional, default: None
        The path to cache folder. If `None`, either ~/.cornac or /tmp/.cornac will be used.

    """

    def __init__(self, url, relative_path, unzip=False, cache_dir=None):
        self.url = url
        self.rel_path = relative_path
        self.unzip = unzip
        self.cache_dir = cache_dir

    def _get_download_path(self):
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.path.expanduser('~'), '.cornac')

        self.cache_dir = os.path.expanduser(self.cache_dir)
        download_path = os.path.join(self.cache_dir, self.rel_path)

        if not os.access(self.cache_dir, os.W_OK):
            self.cache_dir = os.path.join('/tmp', '.cornac')
            download_path = os.path.join(self.cache_dir, self.rel_path)

        if not os.path.exists(os.path.dirname(download_path)):
            os.makedirs(os.path.dirname(download_path))

        return download_path

    def maybe_download(self, verbose=False):
        """Download data if not appearing in cache folder
        """
        fpath = self._get_download_path()

        if os.path.exists(fpath):
            return fpath

        print('Downloading data from', self.url)
        print('and save to', fpath)

        if self.unzip:
            tmp_path = os.path.join(self.cache_dir, 'tmp.zip')
            urlretrieve(self.url, tmp_path)

            if verbose:
                print('Unziping...')
            with zipfile.ZipFile(tmp_path, 'r') as tmp_zip:
                tmp_zip.extractall(self.cache_dir)
            os.remove(tmp_path)
        else:
            urlretrieve(self.url, fpath)

        if verbose:
            print('Downloading finished!')

        return fpath
