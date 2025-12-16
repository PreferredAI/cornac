# Copyright 2023 The Cornac Authors. All Rights Reserved.
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
Yoochoose Dataset is originally from the RecSys Challenge 2015.
"""

from typing import List

from ..data import Reader
from ..utils import cache


def load_buy(fmt="SITJson", reader: Reader = None) -> List:
    """Load the time and location information of check-ins made by users

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, session, item, timestamp, json).
        Location information is stored in `json` format
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/yoochoose/buy.zip",
        unzip=True,
        relative_path="yoochoose/buy.txt",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep="\t")


def load_click(fmt="SITJson", reader: Reader = None) -> List:
    """Load the time and location information of check-ins made by users

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, session, item, timestamp, json).
        Location information is stored in `json` format
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/yoochoose/click.zip",
        unzip=True,
        relative_path="yoochoose/click.txt",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep="\t")


def load_test(fmt="SITJson", reader: Reader = None) -> List:
    """Load the time and location information of check-ins made by users

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, session, item, timestamp, json).
        Location information is stored in `json` format
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/yoochoose/test.zip",
        unzip=True,
        relative_path="yoochoose/test.txt",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep="\t")
