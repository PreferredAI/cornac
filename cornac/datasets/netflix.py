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
"""Link to the data: https://www.kaggle.com/netflix-inc/netflix-prize-data/"""

from typing import List

from ..utils import validate_format
from ..utils import cache
from ..data import Reader

VALID_DATA_FORMATS = ['UIR', 'UIRT']


def _load(fname, fmt='UIR', reader: Reader = None) -> List:
    """Load the Netflix dataset

    Parameters
    ----------
    fname: str, required
        Data file name.

    fmt: str, default: 'UIR'
        Data format to be returned.

    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    fmt = validate_format(fmt, VALID_DATA_FORMATS)
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/netflix/{}.zip'.format(fname),
                  unzip=True, relative_path='netflix/{}.csv'.format(fname))
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt, sep=',')


def load_data(fmt='UIR', reader: Reader = None) -> List:
    """Load the Netflix entire dataset
    - Number of ratings: 100,480,507
    - Number of users:       480,189
    - Number of items:        17,770

    Parameters
    ----------
    fmt: str, default: 'UIR'
        Data format to be returned.

    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    return _load('data', fmt, reader)


def load_data_small(fmt='UIR', reader: Reader = None) -> List:
    """Load a small subset of the Netflix dataset. We draw this subsample such that
    every user has at least 10 items and each item has at least 10 users.
    - Number of ratings: 607,803
    - Number of users:    10,000
    - Number of items:     5,000

    Parameters
    ----------
    fmt: str, default: 'UIR'
        Data format to be returned.

    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    return _load('data_small', fmt, reader)
