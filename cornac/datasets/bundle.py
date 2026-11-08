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
"""Bundle datasets: Youshu, NetEase and iFashion as released by CrossCBR
(https://github.com/mysbupt/CrossCBR), and Spotify as released by AttList
(https://github.com/heyunh2015/AttList). iFashion is the POG subset used by CrossCBR.
"""

from collections import OrderedDict
import json

from ..data import Reader
from ..utils import cache


ARCHIVE_URL = (
    "https://raw.githubusercontent.com/mysbupt/CrossCBR/"
    "bbe80dfb1bbe878d72389185edefc655c8b5db45/dataset.tgz"
)
SPOTIFY_URL = (
    "https://raw.githubusercontent.com/heyunh2015/AttList/"
    "9771fbfe67aa097f81d5a006f61431bbd5c3774e/AttList_cikm2019/data/spotify/"
)
VARIANTS = {
    "youshu": "Youshu",
    "netease": "NetEase",
    "ifashion": "iFashion",
    "spotify": "Spotify",
}
SPLITS = {"train": "train", "validation": "tune", "tune": "tune", "test": "test"}


def _path(variant, filename, cache_dir=None):
    name = VARIANTS.get(variant.lower())
    if name is None:
        raise ValueError("variant must be Youshu, NetEase, iFashion or Spotify")
    if name == "Spotify":
        return cache(
            url=SPOTIFY_URL + filename,
            relative_path="datasets/AttList/Spotify/" + filename,
            cache_dir=cache_dir,
        )
    return cache(
        url=ARCHIVE_URL,
        unzip=True,
        relative_path="datasets/{}/{}".format(name, filename.format(name)),
        cache_dir=cache_dir,
    )


def load_feedback(variant="Youshu", split="train", reader=None, cache_dir=None):
    """Load user-bundle feedback of one of the author splits.

    Parameters
    ----------
    variant: str, default: 'Youshu'
        One of 'Youshu', 'NetEase', 'iFashion' and 'Spotify' (case-insensitive).

    split: str, default: 'train'
        One of 'train', 'validation' (or 'tune') and 'test'.

    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    cache_dir: str, default: None
        Cache directory, Cornac's default directory if None.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, bundle, 1.0).
    """
    split = SPLITS.get(split.lower())
    if split is None:
        raise ValueError("split must be train, validation, tune or test")
    if variant.lower() == "spotify":
        filename = "{}.txt".format("validation" if split == "tune" else split)
        fmt = "UIR"
    else:
        filename = "user_bundle_{}.txt".format(split)
        fmt = "UI"
    fpath = _path(variant, filename, cache_dir)
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep="\t")


def load_user_item(variant="Youshu", reader=None, cache_dir=None):
    """Load user-item feedback. Spotify has none and raises ValueError.

    Parameters
    ----------
    variant: str, default: 'Youshu'
        One of 'Youshu', 'NetEase' and 'iFashion' (case-insensitive).

    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    cache_dir: str, default: None
        Cache directory, Cornac's default directory if None.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, item, 1.0).
    """
    if variant.lower() == "spotify":
        raise ValueError("Spotify has no user-item feedback")
    fpath = _path(variant, "user_item.txt", cache_dir)
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt="UI", sep="\t")


def load_bundle_item(variant="Youshu", reader=None, cache_dir=None):
    """Load bundle-item membership.

    Parameters
    ----------
    variant: str, default: 'Youshu'
        One of 'Youshu', 'NetEase', 'iFashion' and 'Spotify' (case-insensitive).

    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    cache_dir: str, default: None
        Cache directory, Cornac's default directory if None.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (bundle, item).
    """
    reader = Reader() if reader is None else reader
    if variant.lower() == "spotify":
        fpath = _path(variant, "listItem_spotify.txt", cache_dir)
        rows = reader.read(fpath, fmt="UI", sep="\t", parser=_spotify_membership)
        return list(dict.fromkeys((row[0], row[1]) for row in rows))
    fpath = _path(variant, "bundle_item.txt", cache_dir)
    return [(row[0], row[1]) for row in reader.read(fpath, fmt="UI", sep="\t")]


def _spotify_membership(tokens, **kwargs):
    # Each line is a playlist ID and a JSON list of track IDs, possibly repeated.
    return [
        (tokens[0], str(item), 1.0) for item in dict.fromkeys(json.loads(tokens[1]))
    ]


def load_ids(variant="Youshu", cache_dir=None):
    """Load the user, bundle and item catalogs, including IDs without feedback.

    Parameters
    ----------
    variant: str, default: 'Youshu'
        One of 'Youshu', 'NetEase', 'iFashion' and 'Spotify' (case-insensitive).

    cache_dir: str, default: None
        Cache directory, Cornac's default directory if None.

    Returns
    -------
    ids: OrderedDict
        Lists ``user_ids``, ``bundle_ids`` and ``constituent_item_ids`` in numeric
        order, for ``BundleEvaluation.from_splits(..., **load_ids())``.
    """
    keys = ("user_ids", "bundle_ids", "constituent_item_ids")
    if variant.lower() == "spotify":
        with open(_path(variant, "userList_spotify.txt", cache_dir)) as f:
            users = {line.split("\t", 1)[0] for line in f}
        bundles, items = set(), set()
        with open(_path(variant, "listItem_spotify.txt", cache_dir)) as f:
            for line in f:
                bundle_id, tracks = line.rstrip().split("\t", 1)
                bundles.add(bundle_id)
                items.update(str(item) for item in json.loads(tracks))
        catalogs = (users, bundles, items)
        return OrderedDict(zip(keys, (sorted(ids, key=int) for ids in catalogs)))
    with open(_path(variant, "{}_data_size.txt", cache_dir)) as f:
        counts = [int(value) for value in f.readline().split()]
    if len(counts) != 3 or any(count <= 0 for count in counts):
        raise ValueError(
            "{}_data_size.txt must hold three positive counts".format(variant)
        )
    return OrderedDict(zip(keys, ([str(i) for i in range(n)] for n in counts)))
