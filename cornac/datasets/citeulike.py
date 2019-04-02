# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>

This dataset is mostly from the paper 'Collaborative topic modeling for recommending scientific articles'
[Wang and Blei - KDD 2011].  It was further collected, named `citeulike-a`, and used in the paper
'Collaborative Topic Regression with Social Regularization' [Wang, Chen and Li - IJCAI 2013].

Link to the data: http://www.wanghao.in/CDL.htm
"""

from ..utils import cache
from ..data import Reader
from typing import List


def load_data(reader: Reader = None) -> List:
    """Load the implicit feedback between users and items

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, 1).

    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/citeulike/users.zip',
                  relative_path='citeulike/users.dat', unzip=True)
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt='UI', sep=' ', id_inline=True)


def load_text():
    """Load item texts including tile and abstract joined together into one document per item.

    Returns
    -------
    texts: List
        List of text documents, one per item.

    ids: List
        List of item ids aligned with indices in `texts`.
    """
    import csv

    texts, ids = [], []
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/citeulike/text.zip',
                  relative_path='citeulike/raw-data.csv', unzip=True)
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for row in csv.reader(f, delimiter=',', quotechar='"'):
            ids.append(row[0])
            texts.append(row[3] + '. ' + row[4])

    return texts, ids
