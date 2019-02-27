# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>

Data: https://www.kaggle.com/netflix-inc/netflix-prize-data/
"""

from ..utils import validate_format
from ..utils import cache
from ..data import reader

VALID_DATA_FORMATS = ['UIR', 'UIRT']


def load_data(data_format='UIR'):
    """Load the Netflix dataset

    Parameters
    ----------
    data_format: str, default: 'UIR'
        Data format to be returned.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    data_format = validate_format(data_format, VALID_DATA_FORMATS)
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/netflix/data.zip',
                  unzip=True, relative_path='netflix/data.csv')
    if data_format == 'UIR':
        return reader.read_uir(fpath, sep=',')