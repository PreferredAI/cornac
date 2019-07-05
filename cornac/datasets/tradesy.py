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
Link to the data: http://jmcauley.ucsd.edu/data/tradesy/
This data is used in the VBPR paper. After cleaning the data, we have:
- Number of feedback: 394,421 (410,186 is reported but there are duplicates)
- Number of users:     19,243 (19,823 is reported due to duplicates)
- Number of items:    165,906 (166,521 is reported due to duplicates)
"""

from typing import List

import numpy as np

from ..utils import cache
from ..data import Reader
from ..data.reader import read_text


def load_data(reader: Reader = None) -> List:
    """Load the feedback observations

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, 1).

    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/tradesy/users.zip',
                  unzip=True, relative_path='tradesy/users.csv')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt='UI', sep=',')


def load_feature():
    """Load the item visual feature

    Returns
    -------
    features: numpy.ndarray
        Feature matrix with shape (n, 4096) with n is the number of items.

    item_ids: List
        List of item ids aligned with indices in `features`.
    """
    features = np.load(cache(url='https://static.preferred.ai/cornac/datasets/tradesy/item_features.zip',
                             unzip=True, relative_path='tradesy/item_features.npy'))
    item_ids = read_text(cache(url='https://static.preferred.ai/cornac/datasets/tradesy/item_ids.zip',
                               unzip=True, relative_path='tradesy/item_ids.txt'))
    return features, item_ids
