# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>

Original data: http://jmcauley.ucsd.edu/data/tradesy/
This data is used in the VBPR paper:
- Number of users: 92835
- Number of items: 166526
- Number of feedback: 396636 (410186 is reported but there are duplicates)

"""

from ..utils import cache
from ..data import reader

import pickle


def load_data():
    """Load the feedback observations

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item , feedback).

    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/tradesy/data.csv',
                  relative_path='tradesy/data.csv')
    return reader.read_uir(fpath, sep=',', skip_lines=1)


def load_feature():
    """Load the item visual feature

    Returns
    -------
    data: dict
        Item-feature dictionary. Each feature vector is a Numpy array of size 4096.

    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/tradesy/item_feature.pkl',
                  relative_path='tradesy/item_feature.pkl')
    with open(fpath, 'rb') as f:
        return pickle.load(f)
