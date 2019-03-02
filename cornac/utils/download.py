# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import os
import zipfile
from urllib import request
from tqdm import tqdm

def urlretrieve(url, fpath):
    """Retrieve data from given url

    Parameters
    ----------
    url: str
        The url to the data.

    fpath: str
        The path to file where data is stored.

    """
    opener = request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]

    with tqdm(unit='B', unit_scale=True) as progress:
        def report(chunk, chunksize, total):
            progress.total = total
            progress.update(chunksize)
        request.install_opener(opener)
        request.urlretrieve(url, fpath, reporthook=report)


def get_cache_path(relative_path, cache_dir=None):
    """Return the absolute path to the cached data file
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.cornac')
    if not os.access(cache_dir, os.W_OK):
        cache_dir = os.path.join('/tmp', '.cornac')
    cache_path = os.path.join(cache_dir, relative_path)

    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path))

    return cache_path, cache_dir


def cache(url, unzip=False, relative_path=None, cache_dir=None):
    """Download the data and cache to file

    Parameters
    ----------
    url: str
        The url to the data.

    unzip: bool, optional, default: False
        Whether the data is a zip file and going to be unzipped after the download.

    relative_path: str
        Relative path to the data file after finishing the download.
        If unzip=True, relative_path is the path to unzipped file.

    cache_dir: bool, optional, default: None
        The path to cache folder. If `None`, either ~/.cornac or /tmp/.cornac will be used.

    """
    if relative_path is None:
        relative_path = url.split('/')[-1]
    cache_path, cache_dir = get_cache_path(relative_path, cache_dir)
    if os.path.exists(cache_path):
        return cache_path

    print('Data from', url)
    print('will be cached into', cache_path)

    if unzip:
        tmp_path = os.path.join(cache_dir, 'tmp.zip')
        urlretrieve(url, tmp_path)
        print('Unzipping...')
        with zipfile.ZipFile(tmp_path, 'r') as tmp_zip:
            tmp_zip.extractall(cache_dir)
        os.remove(tmp_path)
    else:
        urlretrieve(url, cache_path)

    print('File cached!')
    return cache_path
