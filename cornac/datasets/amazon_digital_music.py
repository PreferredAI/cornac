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
This data is built based on the Amazon datasets
provided by Julian McAuley at: http://jmcauley.ucsd.edu/data/amazon/
"""

from ..utils import cache
from ..data import Reader
from typing import List


def load_feedback(reader: Reader = None) -> List:
    """Load the user-item ratings, scale: [1,5]

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
        url="https://static.preferred.ai/cornac/datasets/amazon_digital_music/rating.zip",
        unzip=True,
        relative_path="amazon_digital_music/rating.txt",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt="UIR", sep=",")


def load_review(reader: Reader = None) -> List:
    """Load the user-item-review list

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, review).
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/amazon_digital_music/review.zip",
        unzip=True,
        relative_path="amazon_digital_music/review.txt",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt="UIReview", sep="\t")
