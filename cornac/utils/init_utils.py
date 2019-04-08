# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np


def zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype=dtype)


def ones(shape, dtype=np.float32):
    return np.ones(shape, dtype=dtype)


def constant(shape, val, dtype=np.float32):
    return ones(shape, dtype=dtype) * val


def uniform(shape, low=0, high=1, seed=None, dtype=np.float32):
    np.random.seed(seed)
    return np.random.uniform(low, high, shape).astype(dtype)


def normal(shape, mean=0, std=1, seed=None, dtype=np.float32):
    np.random.seed(seed)
    return np.random.normal(mean, std, shape).astype(dtype)


def xavier_uniform(shape, seed=None, dtype=np.float32):
    """Return a numpy array by performing 'Xavier' initializer
    also known as 'Glorot' initializer on Uniform distribution.

    References
    ----------
    ** Xavier Glorot and Yoshua Bengio (2010):
    [Understanding the difficulty of training deep feedforward neural networks.
    International conference on artificial intelligence and statistics.]
    (http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    assert len(shape) == 2  # only support matrix
    std = np.sqrt(2.0 / np.sum(shape))
    limit = np.sqrt(3.0) * std
    return uniform(shape, -limit, limit, seed, dtype)


def xavier_normal(shape, seed=None, dtype=np.float32):
    """Return a numpy array by performing 'Xavier' initializer
    also known as 'Glorot' initializer on Normal distribution.

    References
    ----------
    ** Xavier Glorot and Yoshua Bengio (2010):
    [Understanding the difficulty of training deep feedforward neural networks.
    International conference on artificial intelligence and statistics.]
    (http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    assert len(shape) == 2  # only support matrix
    std = np.sqrt(2.0 / np.sum(shape))
    return normal(shape, 0, std, seed, dtype)
