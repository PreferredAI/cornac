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
Diginetica dataset is originally from the CIKM 2016 competition.
"""

from typing import List

from ..data import Reader
from ..utils import cache


def load_train(fmt="USIT", reader: Reader = None) -> List:
    """Load train data

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, session, item, timestamp).
    """
    fpath = cache(
        url="https://static.preferred.ai/cornac/datasets/diginetica/train.zip",
        unzip=True,
        relative_path="diginetica/train.csv",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep=",")


def load_val(fmt="USIT", reader: Reader = None, mode="session-based") -> List:
    """Load validation data

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    mode: str, default: 'session-based'
        - ``'session-based'`` (default): each user's single held-out session
          only (``val_sbr``) — the clean evaluation set for session-based
          models, with no training transitions repeated.
        - ``'session-aware'``: the full cumulative file (``val``), where each
          user's prior sessions precede their held-out session, so a
          session-aware model can use cross-session history at eval time.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, session, item, timestamp).
    """
    if mode not in ("session-based", "session-aware"):
        raise ValueError(f"mode='{mode}' not supported; choose 'session-based' or 'session-aware'")
    name = "val_sbr" if mode == "session-based" else "val"
    fpath = cache(
        url=f"https://static.preferred.ai/cornac/datasets/diginetica/{name}.zip",
        unzip=True,
        relative_path=f"diginetica/{name}.csv",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep=",")


def load_test(fmt="USIT", reader: Reader = None, mode="session-based") -> List:
    """Load test data

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    mode: str, default: 'session-based'
        - ``'session-based'`` (default): each user's single held-out session
          only (``test_sbr``) — the clean evaluation set for session-based
          models, with no training transitions repeated.
        - ``'session-aware'``: the full cumulative file (``test``), where each
          user's prior sessions precede their held-out session, so a
          session-aware model can use cross-session history at eval time.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, session, item, timestamp).
    """
    if mode not in ("session-based", "session-aware"):
        raise ValueError(f"mode='{mode}' not supported; choose 'session-based' or 'session-aware'")
    name = "test_sbr" if mode == "session-based" else "test"
    fpath = cache(
        url=f"https://static.preferred.ai/cornac/datasets/diginetica/{name}.zip",
        unzip=True,
        relative_path=f"diginetica/{name}.csv",
    )
    reader = Reader() if reader is None else reader
    return reader.read(fpath, fmt=fmt, sep=",")
