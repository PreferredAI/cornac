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
This data is built based on the Ta Feng Grocery Dataset that contains 
a Chinese grocery store transaction data from November 2000 to February 2001.
Accessed at https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset
"""

from ..utils import cache
from ..data import Reader
from typing import List


def load_basket(fmt="UBITJson", reader: Reader = None) -> List:
    """Load the transaction data

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, basket, item, timestamp, json).
    """
    fpath = cache(
        url="https://static.preferred.ai/hieudo/basket.zip",
        unzip=True,
        relative_path="tafeng/basket.txt",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep="\t")
