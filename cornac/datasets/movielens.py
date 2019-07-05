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
"""Link to the data: https://grouplens.org/datasets/movielens/"""

from typing import List

from ..utils import validate_format
from ..utils import cache
from ..data import Reader
from ..data.reader import read_text

VALID_DATA_FORMATS = ['UIR', 'UIRT']


def load_100k(fmt='UIR', reader=None):
    """Load the MovieLens 100K dataset

    Parameters
    ----------
    fmt: str, default: 'UIR'
        Data format to be returned.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.

    """
    fmt = validate_format(fmt, VALID_DATA_FORMATS)
    fpath = cache(url='http://files.grouplens.org/datasets/movielens/ml-100k/u.data',
                  relative_path='ml-100k/u.data')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt)


def load_1m(fmt='UIR', reader: Reader = None) -> List:
    """Load the MovieLens 1M dataset

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
    fmt = validate_format(fmt, VALID_DATA_FORMATS)
    fpath = cache(url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                  unzip=True, relative_path='ml-1m/ratings.dat')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt, sep='::')


def load_plot():
    """Load the plots of movies provided @ http://dm.postech.ac.kr/~cartopy/ConvMF/

    Returns
    -------
    texts: List
        List of text documents, one per item.

    ids: List
        List of item ids aligned with indices in `texts`.
    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/movielens/ml_plot.zip',
                  unzip=True, relative_path='movielens/ml_plot.dat')
    texts, ids = read_text(fpath, sep='::')
    return texts, ids
