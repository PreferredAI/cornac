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
"""Link to the dataset: http://www.trustlet.org/downloaded_epinions.html"""

import os
from typing import List

from ..data import Reader
from ..utils import cache
from ..utils.download import get_cache_path


def _get_cache_dir():
    cache_dir = get_cache_path('epinions')[0]
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def load_data(reader: Reader = None) -> List:
    """Load the rating feedback

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).

    """
    fpath = cache(url='http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2',
                  unzip=True, relative_path='ratings_data.txt', cache_dir=_get_cache_dir())
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep=' ')


def load_trust(reader: Reader = None) -> List:
    """Load the trust data

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).

    """
    fpath = cache(url='http://www.trustlet.org/datasets/downloaded_epinions/trust_data.txt.bz2',
                  unzip=True, relative_path='trust_data.txt', cache_dir=_get_cache_dir())
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep=' ')
