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

import ast
import csv
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


def _validate(category: str, version: str) -> None:
    if category not in _CATEGORY_FILES:
        raise ValueError(f"category='{category}' not supported; " f"choose one of {sorted(_CATEGORY_FILES)}")
    if version != "2014":
        raise ValueError(f"version='{version}' not supported; only '2014' (McAuley 5-core) " "is available")


def _reviews_csv(category: str, version: str) -> str:
    stem = _CATEGORY_FILES[category]
    gz_path = cache(
        url=f"{_BASE_URL}/reviews_{stem}_5.json.gz",
        relative_path=f"amazon_review/{category}_{version}.json.gz",
    )
    csv_path = f"{gz_path[:-len('.json.gz')]}.csv"
    if not os.path.exists(csv_path):
        _preprocess(gz_path, csv_path)
    return csv_path


def _item_text(meta: dict, include_description: bool = False) -> str:
    """Flatten item metadata into one text string (title, price, brand,
    categories -- the content features used by TIGER).

    When ``include_description`` is set, the item's ``description`` field (if
    present and non-empty) is appended as a last ``Description: ...`` part.
    """
    parts = []
    title = meta.get("title")
    if title:
        parts.append(f"Title: {title}")
    price = meta.get("price")
    if price is not None:
        parts.append(f"Price: {price}")
    brand = meta.get("brand")
    if brand:
        parts.append(f"Brand: {brand}")
    categories = meta.get("categories")
    if categories:
        flat = []
        for path in categories:
            for cat in path:
                if cat not in flat:
                    flat.append(cat)
        parts.append("Categories: " + ", ".join(flat))
    if include_description:
        description = meta.get("description")
        if description:
            parts.append(f"Description: {description}")
    return ". ".join(" ".join(part.split()) for part in parts)


def _preprocess_meta(meta_gz_path: str, reviews_csv_path: str, out_path: str, include_description: bool = False) -> None:
    """Extract one text string per 5-core item from the raw category metadata.

    The 2014 metadata files contain Python dict literals (not valid JSON),
    hence ``ast.literal_eval``. Items without a metadata entry get an empty
    string so the output covers every item in the reviews file.
    """
    keep = {}  # item id -> insertion order (dict preserves order)
    with open(reviews_csv_path) as f:
        for line in f:
            item = line.split(",")[1]
            if item not in keep:
                keep[item] = len(keep)

    texts = {}
    with gzip.open(meta_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            meta = ast.literal_eval(line)
            asin = meta.get("asin")
            if asin in keep:
                texts[asin] = _item_text(meta, include_description)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        for item in keep:
            writer.writerow([item, texts.get(item, "")])


def load_text(category: str, version: str = "2014", include_description: bool = False) -> (List, List):
    """Load item content text (title, price, brand, categories) per item.

    Texts are built from the public product metadata of the same corpus as
    :func:`load_feedback` and cover exactly the items appearing in the 5-core
    reviews (items without a metadata entry get an empty string). These are
    the content features embedded with Sentence-T5 in the TIGER paper.

    Parameters
    ----------
    category: str, required
        One of ``'beauty'``, ``'sports'``, ``'toys'``.

    version: str, default: '2014'
        Dataset version. Only ``'2014'`` is currently supported.

    include_description: bool, default: False
        If True, append each item's ``description`` field as a last
        ``Description: ...`` part of its text. Paischer et al.
        (arXiv:2412.08604) found this beneficial for the Toys dataset, while
        attribute-only text works better for Beauty/Sports. Descriptions are
        kept in full (downstream sentence encoders truncate as needed). The
        description variant is cached separately (``*_text_desc.csv``) so it
        never overwrites the attribute-only cache.

    Returns
    -------
    texts: List
        List of text documents, one per item.

    ids: List
        List of item ids aligned with indices in `texts`.
    """
    _validate(category, version)
    reviews_csv_path = _reviews_csv(category, version)
    suffix = "_text_desc" if include_description else "_text"
    text_path = f"{reviews_csv_path[:-len('.csv')]}{suffix}.csv"
    if not os.path.exists(text_path):
        stem = _CATEGORY_FILES[category]
        meta_gz_path = cache(
            url=f"{_BASE_URL}/meta_{stem}.json.gz",
            relative_path=f"amazon_review/meta_{category}_{version}.json.gz",
        )
        _preprocess_meta(meta_gz_path, reviews_csv_path, text_path, include_description)

    texts, ids = [], []
    with open(text_path, newline="") as f:
        for item, text in csv.reader(f):
            ids.append(item)
            texts.append(text)
    return texts, ids


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
    _validate(category, version)
    csv_path = _reviews_csv(category, version)

    reader = Reader() if reader is None else reader
    return reader.read(csv_path, fmt=fmt, sep=",")
