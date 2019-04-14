# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>

Link to the dataset: http://www.trustlet.org/downloaded_epinions.html
"""

import os
from typing import List

from ..data import Reader
from ..utils import cache
from ..utils.download import get_cache_path


def _get_cache_dir():
    cache_dir = get_cache_path('epinions')[0]
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def load_data(reader: Reader = None) -> List:
    """Load the rating feedback

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).

    """
    fpath = cache(url='http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2',
                  unzip=True, relative_path='ratings_data.txt', cache_dir=_get_cache_dir())
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep=' ')


def load_trust(reader: Reader = None) -> List:
    """Load the trust data

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).

    """
    fpath = cache(url='http://www.trustlet.org/datasets/downloaded_epinions/trust_data.txt.bz2',
                  unzip=True, relative_path='trust_data.txt', cache_dir=_get_cache_dir())
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep=' ')
