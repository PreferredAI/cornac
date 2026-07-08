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
Amazon Product Review datasets.

There are three versions: '2014', '2018', and '2023' available.
'2014' is the version used in the Semantic-ID literature (e.g., TIGER).

Source: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
"""

import gzip
import json
import os
from typing import List

from ..data import Reader
from ..utils import cache

# category -> reviews_<cat>_5.json.gz
_CATEGORY_FILES = {
    "beauty": "Beauty",
    "sports": "Sports_and_Outdoors",
    "toys": "Toys_and_Games",
}

_BASE_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"


def _preprocess(gz_path: str, csv_path: str) -> None:
    """Parse the raw 5-core reviews into ``user,item,rating,timestamp`` rows.

    Only mechanical cleaning is applied (drop rows with a missing field);
    rows are sorted chronologically per user so downstream sequential builders
    receive time-ordered sessions.
    """
    rows = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user = r.get("reviewerID")
            item = r.get("asin")
            rating = r.get("overall")
            timestamp = r.get("unixReviewTime")
            if user is None or item is None or rating is None or timestamp is None:
                continue
            rows.append((user, item, float(rating), int(timestamp)))

    rows.sort(key=lambda x: (x[0], x[3]))  # (user, timestamp)

    with open(csv_path, "w") as f:
        for user, item, rating, timestamp in rows:
            f.write(f"{user},{item},{rating},{timestamp}\n")


def load_feedback(category: str, version: str = "2014", fmt: str = "UIRT", reader: Reader = None) -> List:
    """Load the user-item review feedback, chronologically ordered per user.

    Parameters
    ----------
    category: str, required
        One of ``'beauty'``, ``'sports'``, ``'toys'`` -- the three categories
        used by TIGER and subsequent Semantic-ID papers.

    version: str, default: '2014'
        Dataset version. Only ``'2014'`` (McAuley 5-core) is currently supported;
        2018 and 2023 are available, but 2014 is the version used throughout the Semantic-ID literature.

    fmt: str, default: 'UIRT'
        Data format; the underlying file has user, item, rating, and timestamp
        columns, so ``'UIR'`` and ``'UI'`` are also valid.

    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, rating, timestamp).
    """
    if category not in _CATEGORY_FILES:
        raise ValueError(f"category='{category}' not supported; " f"choose one of {sorted(_CATEGORY_FILES)}")
    if version != "2014":
        raise ValueError(f"version='{version}' not supported; only '2014' (McAuley 5-core) " "is available")

    stem = _CATEGORY_FILES[category]
    gz_path = cache(
        url=f"{_BASE_URL}/reviews_{stem}_5.json.gz",
        relative_path=f"amazon_review/{category}_{version}.json.gz",
    )
    csv_path = f"{gz_path[:-len('.json.gz')]}.csv"
    if not os.path.exists(csv_path):
        _preprocess(gz_path, csv_path)

    reader = Reader() if reader is None else reader
    return reader.read(csv_path, fmt=fmt, sep=",")
