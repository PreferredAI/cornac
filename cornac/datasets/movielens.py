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
from collections import namedtuple

from ..utils import validate_format
from ..utils import cache
from ..data import Reader
from ..data.reader import read_text


VALID_DATA_FORMATS = ["UIR", "UIRT"]

MovieLens = namedtuple("MovieLens", ["url", "unzip", "path", "sep", "skip"])
ML_DATASETS = {
    "100K": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-100k/u.data",
        False,
        "ml-100k/u.data",
        "\t",
        0,
    ),
    "1M": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        True,
        "ml-1m/ratings.dat",
        "::",
        0,
    ),
    "10M": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
        True,
        "ml-10M100K/ratings.dat",
        "::",
        0,
    ),
    "20M": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        True,
        "ml-20m/ratings.csv",
        ",",
        1,
    ),
}


def load_feedback(fmt="UIR", variant="100K", reader=None):
    """Load the user-item ratings of one of the MovieLens datasets

    Parameters
    ----------
    fmt: str, default: 'UIR'
        Data format to be returned, one of ['UIR', 'UIRT'].

    variant: str, optional, default: '100K'
        Specifies which MovieLens dataset to load, one of ['100K', '1M', '10M', '20M'].

    reader: `obj:cornac.data.Reader`, optional, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples depending on the given data format.
    """

    fmt = validate_format(fmt, VALID_DATA_FORMATS)

    ml = ML_DATASETS.get(variant.upper(), None)
    if ml is None:
        raise ValueError("variant must be one of {}.".format(ML_DATASETS.keys()))

    fpath = cache(url=ml.url, unzip=ml.unzip, relative_path=ml.path)
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt, sep=ml.sep, skip_lines=ml.skip)


def load_plot():
    """Load the plots of movies provided @ http://dm.postech.ac.kr/~cartopy/ConvMF/

    Returns
    -------
    texts: List
        List of text documents, one per item.

    ids: List
        List of item ids aligned with indices in `texts`.
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/movielens/ml_plot.zip",
        unzip=True,
        relative_path="movielens/ml_plot.dat",
    )
    texts, ids = read_text(fpath, sep="::")
    return texts, ids
