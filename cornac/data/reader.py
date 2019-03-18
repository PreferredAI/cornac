# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import itertools


def read_uir(fpath, u_col=0, i_col=1, r_col=2, sep='\t', skip_lines=0):
    """Read data in the form of triplets (user, item, rating).

    Parameters
    ----------
    fpath: str
        Path to the data file

    u_col: int, default: 0
        Index of the user column

    i_col: int, default: 1
        Index of the item column

    r_col: int, default: 2
        Index of the rating column

    sep: str, default: \t
        The delimiter string.

    skip_lines: int, default: 0
        Number of first lines to skip

    Returns
    -------
    triplets: :obj:`iterable`
        Data in the form of list of tuples of (user, item, rating).

    """
    triplets = []
    with open(fpath, 'r') as f:
        for line in itertools.islice(f, skip_lines, None):
            tokens = [token.strip() for token in line.split(sep)]
            triplets.append((tokens[u_col], tokens[i_col], float(tokens[r_col])))
    return triplets


def read_ui(fpath, value=1.0, sep='\t', skip_lines=0):
    """Read data in the form of implicit feedback user-items.
    Each line starts with user id followed by multiple of item ids.

    Parameters
    ----------
    fpath: str
        Path to the data file

    value: float, default: 1.0
        Value for the feedback

    sep: str, default: \t
        The delimiter string.

    skip_lines: int, default: 0
        Number of first lines to skip

    Returns
    -------
    triplets: :obj:`iterable`
        Data in the form of list of tuples of (user, item, 1).

    """
    triplets = []
    with open(fpath, 'r') as f:
        for line in itertools.islice(f, skip_lines, None):
            tokens = [token.strip() for token in line.split(sep)]
            triplets.extend([tokens[0], iid, value] for iid in tokens[1:])
    return triplets