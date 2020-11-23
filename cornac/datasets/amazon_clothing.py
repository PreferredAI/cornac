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
This data is built based on the Amazon datasets provided by Julian McAuley @ http://jmcauley.ucsd.edu/data/amazon/.
We make sure all items having three types of auxiliary data: text, image, and context (items appearing together).
"""

from typing import List

import numpy as np

from ..utils import cache
from ..data import Reader
from ..data.reader import read_text


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
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/amazon_clothing/rating.zip',
                  unzip=True, relative_path='amazon_clothing/rating.txt')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep='\t')


def load_text():
    """Load the item text descriptions

    Returns
    -------
    texts: List
        List of text documents, one per item.

    ids: List
        List of item ids aligned with indices in `texts`.
    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/amazon_clothing/text.zip',
                  unzip=True, relative_path='amazon_clothing/text.txt')
    texts, ids = read_text(fpath, sep='::')
    return texts, ids


def load_visual_feature():
    """Load item visual features (extracted from pre-trained CNN)

    Returns
    -------
    features: numpy.ndarray
        Feature matrix with shape (n, 4096) with n is the number of items.

    item_ids: List
        List of item ids aligned with indices in `features`.
    """
    features = np.load(cache(url='https://static.preferred.ai/cornac/datasets/amazon_clothing/image.zip',
                             unzip=True, relative_path='amazon_clothing/image_features.npy'))
    item_ids = read_text(cache(url='https://static.preferred.ai/cornac/datasets/amazon_clothing/item_ids.zip',
                               unzip=True, relative_path='amazon_clothing/item_ids.txt'))
    return features, item_ids


def load_graph(reader: Reader = None) -> List:
    """Load the item-item interactions (symmetric network), built from the Amazon Also-Viewed information

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (item, item, 1).
    """
    fpath = cache(url='https://static.preferred.ai/cornac/datasets/amazon_clothing/context.zip',
                  unzip=True, relative_path='amazon_clothing/context.txt')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt='UI', sep='\t')
