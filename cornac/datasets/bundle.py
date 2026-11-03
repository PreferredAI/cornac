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
"""Author-hosted Youshu, NetEase, iFashion and AttList Spotify bundle data.

Sources: https://github.com/mysbupt/CrossCBR (Ma et al., KDD2022).
Spotify: https://github.com/heyunh2015/AttList (He et al., CIKM2019).
These public releases have no declared data license; availability does not grant
an open-data license. iFashion is the processed POG subset used by CrossCBR.
"""

from collections import OrderedDict
import json

from ..data import Reader
from ..utils import cache


ARCHIVE_URL = ("https://raw.githubusercontent.com/mysbupt/CrossCBR/"
               "bbe80dfb1bbe878d72389185edefc655c8b5db45/dataset.tgz")
ARCHIVE_SHA256 = "95dcb76dec19ba41601adaa03e290b67ca5b7a9b5d36ff4b07b5553ffa0b1be2"
SPOTIFY_URL = ("https://raw.githubusercontent.com/heyunh2015/AttList/"
               "9771fbfe67aa097f81d5a006f61431bbd5c3774e/AttList_cikm2019/data/spotify/")
_VARIANTS = {"youshu": "Youshu", "netease": "NetEase", "ifashion": "iFashion", "spotify": "Spotify"}


def _path(variant, filename, cache_dir=None):
    canonical = _VARIANTS.get(variant.lower())
    if canonical is None:
        raise ValueError("variant must be Youshu, NetEase, iFashion or Spotify")
    if canonical == "Spotify":
        return cache(url=SPOTIFY_URL + filename,
                     relative_path="datasets/AttList/Spotify/" + filename,
                     cache_dir=cache_dir)
    return cache(url=ARCHIVE_URL, unzip=True,
                 relative_path="datasets/{}/{}".format(canonical, filename.format(canonical)),
                 cache_dir=cache_dir)


def load_feedback(variant="Youshu", split="train", reader=None, cache_dir=None):
    """Load author user-bundle splits as (string user, string bundle, 1.0).

    ``variant`` is case-insensitive Youshu, NetEase, iFashion or Spotify. ``split`` is
    train, validation (alias tune), or test. An optional Cornac Reader applies
    ordinary filtering. The pinned author archive is downloaded/cached lazily. ``cache_dir`` optionally
    selects a cache root instead of Cornac's default directory.
    """
    split = {"train": "train", "validation": "tune", "tune": "tune", "test": "test"}.get(split.lower())
    if split is None:
        raise ValueError("split must be train, validation, tune or test")
    filename = ("{}.txt".format("validation" if split == "tune" else split)
                if variant.lower() == "spotify" else "user_bundle_{}.txt".format(split))
    path = _path(variant, filename, cache_dir)
    fmt = "UIR" if variant.lower() == "spotify" else "UI"
    return (Reader() if reader is None else reader).read(path, fmt=fmt, sep="\t")


def load_user_item(variant="Youshu", reader=None, cache_dir=None):
    """Load independent user-item feedback; Spotify has none and raises ValueError."""
    if variant.lower() == "spotify":
        raise ValueError("Spotify has no independent user-item feedback; neural bundle models "
                         "require independent UI data. UI is not derived from playlists.")
    path = _path(variant, "user_item.txt", cache_dir)
    return (Reader() if reader is None else reader).read(path, fmt="UI", sep="\t")


def load_bundle_item(variant="Youshu", reader=None, cache_dir=None):
    """Load (string bundle, string constituent item) membership pairs."""
    if variant.lower() == "spotify":
        path = _path(variant, "listItem_spotify.txt", cache_dir)
        rows = (Reader() if reader is None else reader).read(
            path, fmt="UI", sep="\t", parser=_spotify_membership)
        return list(dict.fromkeys((row[0], row[1]) for row in rows))
    path = _path(variant, "bundle_item.txt", cache_dir)
    rows = (Reader() if reader is None else reader).read(path, fmt="UI", sep="\t")
    return [(row[0], row[1]) for row in rows]


def _spotify_membership(tokens, **kwargs):
    return [(tokens[0], str(item), 1.0) for item in dict.fromkeys(json.loads(tokens[1]))]


def load_ids(variant="Youshu", cache_dir=None):
    """Return ordered catalog lists including unobserved IDs from author metadata.

    The OrderedDict keys are user_ids, bundle_ids and constituent_item_ids,
    suitable for ``BundleEvaluation.from_splits(..., **load_ids())``. Numeric
    archive IDs are strings in numeric order (Spotify tracks start at 1). Explicit catalogs preserve
    UI-only users and reference token-truncation order.
    """
    if variant.lower() == "spotify":
        path = _path(variant, "userList_spotify.txt", cache_dir)
        with open(path, encoding="utf-8") as stream:
            users = {line.split("\t", 1)[0] for line in stream}
        path = _path(variant, "listItem_spotify.txt", cache_dir)
        bundles, items = set(), set()
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                bundle_id, contents = line.rstrip().split("\t", 1)
                bundles.add(bundle_id)
                items.update(str(item) for item in json.loads(contents))
        return OrderedDict((key, sorted(ids, key=int)) for key, ids in zip(
            ("user_ids", "bundle_ids", "constituent_item_ids"), (users, bundles, items)))
    path = _path(variant, "{}_data_size.txt", cache_dir)
    with open(path, encoding="utf-8") as stream:
        counts = [int(value) for value in stream.readline().split()]
    if len(counts) != 3 or any(count <= 0 for count in counts):
        raise ValueError("bundle metadata must contain three positive catalog sizes")
    return OrderedDict((key, [str(index) for index in range(count)])
                       for key, count in zip(("user_ids", "bundle_ids", "constituent_item_ids"), counts))
