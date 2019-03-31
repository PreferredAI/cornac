# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>

MovieLens: https://grouplens.org/datasets/movielens/
"""

from ..utils import validate_format
from ..utils import cache
from ..data import reader

VALID_DATA_FORMATS = ['UIR', 'UIRT']


def load_100k(fmt='UIR'):
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
    if fmt == 'UIR':
        return reader.read_uir(fpath)


def load_1m(fmt='UIR'):
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
    if fmt == 'UIR':
        return reader.read_uir(fpath, sep='::')


def load_plot():
    """Load the plots of movies provided @ http://dm.postech.ac.kr/~cartopy/ConvMF/

    Returns
    -------
    plots: List
        A dictionary with keys are movie ids and values are text plots.

    movie_ids: List
        List of ids aligned with indices in `plots`.
    """
    plots, movie_ids = [], []
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/movielens/ml_plot.zip',
                  unzip=True, relative_path='movielens/ml_plot.dat')
    with open(fpath, 'r') as f:
        for line in f:
            movie_id, plot = line.strip().split('::')
            movie_ids.append(movie_id)
            plots.append(plot)
    return plots, movie_ids
