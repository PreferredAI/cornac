# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numbers

import numpy as np


def sigmoid(x):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-x))


def scale(values, target_min, target_max, source_min=None, source_max=None):
    """Scale the value of a numpy array "values"
    from source_min, source_max into a range [target_min, target_max]

    Parameters
    ----------
    values : Numpy array, required
        Values to be scaled.

    target_min : scalar, required
        Target minimum value.

    target_max : scalar, required
        Target maximum value.

    source_min : scalar, required, default: None
        Source minimum value if desired. If None, it will be the minimum of values.

    source_max : scalar, required, default: None
        Source minimum value if desired. If None, it will be the maximum of values.

    Returns
    -------
    res: Numpy array
        Output values mapped into range [target_min, target_max]
    """
    if source_min is None:
        source_min = np.min(values)
    if source_max is None:
        source_max = np.max(values)
    if source_min == source_max:  # improve this scenario
        source_min = 0.0

    values = (values - source_min) / (source_max - source_min)
    values = values * (target_max - target_min) + target_min
    return values


def clip(values, lower_bound, upper_bound):
    """Perform clipping to enforce values to lie
    in a specific range [lower_bound, upper_bound]

    Parameters
    ----------
    values : Numpy array, required
        Values to be clipped.

    lower_bound : scalar, required
        Lower bound of the output.

    upper_bound : scalar, required
        Upper bound of the output.

    Returns
    -------
    res: Numpy array
        Clipped values in range [lower_bound, upper_bound]
    """
    values = np.where(values > upper_bound, upper_bound, values)
    values = np.where(values < lower_bound, lower_bound, values)

    return values


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


def validate_format(input_format, valid_formats):
    """Check the input format is in list of valid formats
    :raise ValueError if not supported
    """
    if not input_format in valid_formats:
        raise ValueError('{} data format is not in valid formats ({})'.format(input_format, valid_formats))

    return input_format


def estimate_batches(input_size, batch_size):
    """
    Estimate number of batches give `input_size` and `batch_size`
    """
    return int(np.ceil(input_size / batch_size))


def get_rng(seed):
    '''Return a RandomState of Numpy.
    If seed is None, use RandomState singleton from numpy.
    If seed is an integer, create a RandomState from that seed.
    If seed is already a RandomState, just return it.
    '''
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} can not be used to create a numpy.random.RandomState'.format(seed))
