# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>

This data is built based on the Amazon datasets provided by Julian McAuley at: http://jmcauley.ucsd.edu/data/amazon/
"""

from ..utils import validate_format
from ..utils import cache
from ..data import reader

VALID_DATA_FORMATS = ['UIR', 'UIRT']


