# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>

MovieLens: https://grouplens.org/datasets/movielens/
"""

from ..utils import validate_format
from ..utils import cache
from ..data import Reader

VALID_DATA_FORMATS = ['UIR', 'UIRT']


def load_100k(fmt='UIR', reader=None):
    """Load the MovieLens 100K dataset

    Parameters
    ----------
    fmt: str, default: 'UIR'
        Data format to be returned.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    fmt = validate_format(fmt, VALID_DATA_FORMATS)
    fpath = cache(url='http://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                  relative_path='ml-100k/u.data')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt)


def load_1m(fmt='UIR', reader=None):
    """Load the MovieLens 1M dataset

    Parameters
    ----------
    fmt: str, default: 'UIR'
        Data format to be returned.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    fmt = validate_format(fmt, VALID_DATA_FORMATS)
    fpath = cache(url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                  relative_path='ml-1m/ratings.dat', unzip=True)
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt, sep='::')


def load_plot():
    """Load the plots of movies provided @ http://dm.postech.ac.kr/~cartopy/ConvMF/

    Returns
    -------
    texts: List
        List of text documents, one per item.

    ids: List
        List of item ids aligned with indices in `texts`.
    """
    texts, ids = [], []
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/movielens/ml_plot.zip',
                  relative_path='movielens/ml_plot.dat', unzip=True)
    with open(fpath, 'r') as f:
        for line in f:
            movie_id, plot = line.strip().split('::')
            texts.append(plot)
            ids.append(movie_id)

    return texts, ids
