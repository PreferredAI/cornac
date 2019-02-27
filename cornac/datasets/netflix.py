# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>

Data: https://www.kaggle.com/netflix-inc/netflix-prize-data/
"""

from ..utils import validate_format
from ..utils import cache
from ..data import reader

VALID_DATA_FORMATS = ['UIR', 'UIRT']


def _load(data_file, data_format='UIR'):
    """Load the Netflix dataset

    Parameters
    ----------
    data_file: str, required
        Data file name.

    data_format: str, default: 'UIR'
        Data format to be returned.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    data_format = validate_format(data_format, VALID_DATA_FORMATS)
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/netflix/{}.zip'.format(data_file),
                  unzip=True, relative_path='netflix/{}.csv'.format(data_file))
    if data_format == 'UIR':
        return reader.read_uir(fpath, sep=',')


def load_data(data_format='UIR'):
    """Load the Netflix entire dataset
        - Number of ratings: 100,480,507
        - Number of users:       480,189
        - Number of items:        17,770

    Parameters
    ----------
    data_format: str, default: 'UIR'
        Data format to be returned.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    return _load('data', data_format)


def load_data_small(data_format='UIR'):
    """Load a small subset of the Netflix dataset. We draw this subsample such that
    every user has at least 10 items and each item has at least 10 users.
        - Number of ratings: 607,803
        - Number of users:    10,000
        - Number of items:     5,000

    Parameters
    ----------
    data_format: str, default: 'UIR'
        Data format to be returned.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    return _load('data_small', data_format)