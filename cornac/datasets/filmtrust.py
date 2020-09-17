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
"""
Source: https://www.librec.net/datasets.html
"""
import os
from typing import List

from ..utils import cache
from ..data import Reader

from ..utils.download import get_cache_path


def _get_cache_dir():
    cache_dir = get_cache_path("filmtrust")[0]
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def load_feedback(reader: Reader = None) -> List:
    """Load the user-item ratings, scale: [0.5,4]

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating).
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/filmtrust/filmtrust.zip",
        unzip=True,
        cache_dir=_get_cache_dir(),
        relative_path="ratings.txt",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep=" ")


def load_trust(reader: Reader = None) -> List:
    """Load the user-user trust information (undirected network)

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, user, 1).
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/filmtrust/filmtrust.zip",
        unzip=True,
        cache_dir=_get_cache_dir(),
        relative_path="trust.txt",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep=" ")
