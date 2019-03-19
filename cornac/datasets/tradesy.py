# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>

Original data: http://jmcauley.ucsd.edu/data/tradesy/
This data is used in the VBPR paper. After cleaning the data, we have:
- Number of feedback: 394,421 (410,186 is reported but there are duplicates)
- Number of users:     19,243 (19,823 is reported due to duplicates)
- Number of items:    165,906 (166,521 is reported due to duplicates)

"""

from ..utils import cache
from ..data import reader


def load_data():
    """Load the feedback observations

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, 1).

    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/tradesy/users.zip',
                  unzip=True, relative_path='tradesy/users.csv')
    return reader.read_ui(fpath, sep=',')


def load_feature():
    """Load the item visual feature

    Returns
    -------
    data: dict
        Item-feature dictionary. Each feature vector is a Numpy array of size 4096.

    """
    import pickle

    fpath = cache(url='https://static.preferred.ai/cornac/datasets/tradesy/item_feature.zip',
                  unzip=True, relative_path='tradesy/item_feature.pkl')
    with open(fpath, 'rb') as f:
        return pickle.load(f)
