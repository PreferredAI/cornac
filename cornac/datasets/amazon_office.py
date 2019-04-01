# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>

This data is built based on the Amazon datasets provided by Julian McAuley at: http://jmcauley.ucsd.edu/data/amazon/
"""

from ..utils import cache
from ..data import reader


def load_rating():
    """Load the user-item ratings

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).
    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/amazon_office/rating.zip',
                  relative_path='amazon_office/rating.txt', unzip=True)
    return reader.read_uir(fpath, sep=' ')


def load_context():
    """Load the item-item interactions

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (item, item, 1).
    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/amazon_office/context.zip',
                  relative_path='amazon_office/context.txt', unzip=True)
    return reader.read_uir(fpath, sep=' ')
