# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>

This data is built based on the Amazon datasets provided by Julian McAuley at: http://jmcauley.ucsd.edu/data/amazon/
"""

from ..utils import validate_format
from ..utils import cache
from ..data import reader

VALID_DATA_FORMATS = ['UIR', 'UIRT']


def load_rating(data_format='UIR'):
    """Load the user-item ratings

    Parameters
    ----------
    data_format: str, default: 'UIR'
        Data format to be returned.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the specified data format.
    """

    data_format = validate_format(data_format, VALID_DATA_FORMATS)
    fpath = cache(url='',
                  relative_path='amazon_office/rating.txt')
    if data_format == 'UIR':
        return reader.read_uir(fpath)