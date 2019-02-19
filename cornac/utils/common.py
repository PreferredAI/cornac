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


def sigmoid(x):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-x))


def map_to(x, t_min, t_max, o_min=None, o_max=None):
    """Map the value of a numpy array "x"
    from o_min, o_max into a range[t_min,t_max]
    """
    if o_min is None:
        o_min = np.min(x)
    if o_max is None:
        o_max = np.max(x)

    return ((x - o_min) / (o_max - o_min)) * (t_max - t_min) + t_min


def clipping(x, min_, max_):
    """Perform clipping to enforce values to lie
    in a specific range [min_,max_]
    """
    x = np.where(x > max_, max_, x)
    x = np.where(x < min_, min_, x)
    return x


def intersects(x, y, assume_unique=False):
    """Return the intersection of given two arrays
    """
    mask = np.in1d(x, y, assume_unique=assume_unique)
    x_intersects_y = x[mask]
    return x_intersects_y


def excepts(x, y, assume_unique=False):
    """Removing elements in array y from array x
    """
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


def validate_data_format(data_format, valid_formats):
    """Check the input data format is supported or not
        - UIR: (user, item, rating) triplet data
        - UIRT: (user, item , rating, timestamp) quadruplet data

    :raise ValueError if not supported
    """
    data_format = str(data_format).upper()
    valid_formats = [str(fmt).upper() for fmt in valid_formats]
    if not data_format in valid_formats:
        raise ValueError('{} data format is not in valid formats ({})'.format(data_format, valid_formats))

    return data_format
