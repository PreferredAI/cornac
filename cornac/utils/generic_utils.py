# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
         Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""



import operator
import numpy as np


def which_(a, op, x):
    ops = {'>': operator.gt,
           '<': operator.lt,
           '>=': operator.ge,
           '<=': operator.le,
           '==': operator.eq,
           '!=': operator.ne}

    i = np.array(range(0, len(a)))

    return i[ops[op](a, x)]


# Sigmoid function
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# Map the value of a numpy array "x" from o_min, o_max into a range[t_min,t_max]
def map_to(x, t_min, t_max, o_min=None, o_max=None):
    if o_min is None:
        o_min = np.min(x)
    if o_max is None:
        o_max = np.max(x)

    return ((x - o_min) / (o_max - o_min)) * (t_max - t_min) + t_min


# Perform clipping to enforce values to lie in a specific range [min_,max_]
def clipping(x, min_, max_):
    x = np.where(x > max_, max_, x)
    x = np.where(x < min_, min_, x)
    return x


def intersects(x, y, assume_unique=False):
    mask = np.in1d(x, y, assume_unique=assume_unique)
    x_intersects_y = x[mask]
    return x_intersects_y


def excepts(x, y, assume_unique=False):
    mask = np.in1d(x, y, assume_unique=assume_unique, invert=True)
    x_excepts_y = x[mask]
    return x_excepts_y


def safe_indexing(X, indices):
    """Return items or rows from X using indices.
    Allows simple indexing of lists or arrays.
    Borrowed from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py

    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.

    Returns
    -------
    subset
        Subset of X on first axis
    """
    if hasattr(X, "shape"):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]


def validate_data_format(data_format):
    """Check the input data format is supported or not
        - UIR: (user, item, rating) triplet data
        - UIRT: (user, item , rating, timestamp) quadruplet data

    :raise ValueError if not supported
    """
    data_format = str(data_format).upper()
    if not data_format in ['UIR', 'UIRT']:
        raise ValueError('{} data format is not supported!'.format(data_format))

    return data_format
