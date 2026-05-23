# Copyright 2026 The Cornac Authors. All Rights Reserved.
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
RetailRocket dataset: e-commerce web events (clicks, add to carts, transactions) data for 4.5 months.
"""

from typing import List

from ..data import Reader
from ..utils import cache


def load_train(fmt="USIT", reader: Reader = None) -> List:
    """Load train data

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, session, item, timestamp).
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/retailrocket/train.zip",
        unzip=True,
        relative_path="retailrocket/train.csv",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep=",")


def load_val(fmt="USIT", reader: Reader = None) -> List:
    """Load validation data

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, session, item, timestamp).
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/retailrocket/val.zip",
        unzip=True,
        relative_path="retailrocket/val.csv",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep=",")


def load_test(fmt="USIT", reader: Reader = None) -> List:
    """Load test data

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, session, item, timestamp).
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/retailrocket/test.zip",
        unzip=True,
        relative_path="retailrocket/test.csv",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep=",")
