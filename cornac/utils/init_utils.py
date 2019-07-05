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

import numpy as np

from .common import get_rng


def zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype=dtype)


def ones(shape, dtype=np.float32):
    return np.ones(shape, dtype=dtype)


def constant(shape, val, dtype=np.float32):
    return ones(shape, dtype=dtype) * val


def uniform(shape=None, low=0.0, high=1.0, random_state=None, dtype=np.float32):
    """
    Draw samples from a uniform distribution.

    Parameters
    ----------
    shape : int or tuple of ints, optional
        Output shape. If shape is ``None`` (default), a single value is returned.
    low : float or array_like of floats, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float or array_like of floats
        Upper boundary of the output interval.  All values generated will be
        less than high.  The default value is 1.0.
    random_state : int or np.random.RandomState, optional
        If an integer is given, it will be used as seed value for creating a RandomState.
    dtype : str or dtype
        Returned data-type for the output array.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized uniform distribution.
    """
    return get_rng(random_state).uniform(low, high, shape).astype(dtype)


def normal(shape=None, mean=0.0, std=1.0, random_state=None, dtype=np.float32):
    """
    Draw random samples from a normal (Gaussian) distribution.

    Parameters
    ----------
    shape : int or tuple of ints, optional
        Output shape. If shape is ``None`` (default), a single value is returned.
    mean : float or array_like of floats
        Mean of the distribution.
    std : float or array_like of floats
        Standard deviation of the distribution.
    random_state : int or np.random.RandomState, optional
        If an integer is given, it will be used as seed value for creating a RandomState.
    dtype : str or dtype
        Returned data-type for the output array.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized normal distribution.
    """
    return get_rng(random_state).normal(mean, std, shape).astype(dtype)


def xavier_uniform(shape, random_state=None, dtype=np.float32):
    """Return a numpy array by performing 'Xavier' initializer
    also known as 'Glorot' initializer on Uniform distribution.

    Parameters
    ----------
    shape : int or tuple of ints
        Output shape.
    random_state : int or np.random.RandomState, optional
        If an integer is given, it will be used as seed value for creating a RandomState.
    dtype : str or dtype
        Returned data-type for the output array.

    Returns
    -------
    out : ndarray
        Output matrix.

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
    return uniform(shape, -limit, limit, random_state, dtype)


def xavier_normal(shape, random_state=None, dtype=np.float32):
    """Return a numpy array by performing 'Xavier' initializer
    also known as 'Glorot' initializer on Normal distribution.

    Parameters
    ----------
    shape : int or tuple of ints
        Output shape.
    random_state : int or np.random.RandomState, optional
        If an integer is given, it will be used as seed value for creating a RandomState.
    dtype : str or dtype
        Returned data-type for the output array.

    Returns
    -------
    out : ndarray
        Output matrix.

    References
    ----------
    ** Xavier Glorot and Yoshua Bengio (2010):
    [Understanding the difficulty of training deep feedforward neural networks.
    International conference on artificial intelligence and statistics.]
    (http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    assert len(shape) == 2  # only support matrix
    std = np.sqrt(2.0 / np.sum(shape))
    return normal(shape, 0, std, random_state, dtype)
