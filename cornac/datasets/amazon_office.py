# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>

This data is built based on the Amazon datasets provided by Julian McAuley at: http://jmcauley.ucsd.edu/data/amazon/
"""

from ..utils import cache
from ..data import Reader


def load_rating(reader=None):
    """Load the user-item ratings

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).
    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/amazon_office/rating.zip',
                  unzip=True, relative_path='amazon_office/rating.txt')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep=' ')


def load_context(reader=None):
    """Load the item-item interactions

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (item, item, 1).
    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/amazon_office/context.zip',
                  unzip=True, relative_path='amazon_office/context.txt')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep=' ')
