# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..utils.download import DownloadItem
from ..data import reader


def load_100k(format='UIR', verbose=False):
    """Load the MovieLens 100K dataset

    Parameters
    ----------
    format: str, default: 'UIR'
        Data format to be returned.

    verbose: bool, default: False
        The verbosity flag.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    fpath = DownloadItem(url='http://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                         relative_path='ml-100k/u.data').maybe_download(verbose)
    if format == 'UIR':
        return reader.read_uir(fpath)


def load_1m(format='UIR', verbose=False):
    """Load the MovieLens 1M dataset

    Parameters
    ----------
    format: str, default: 'UIR'
        Data format to be returned.

    verbose: bool, default: False
        The verbosity flag.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    fpath = DownloadItem(url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                         relative_path='ml-1m/ratings.dat', unzip=True).maybe_download(verbose)
    if format == 'UIR':
        return reader.read_uir(fpath, sep='::')
