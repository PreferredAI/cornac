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
"""Shared utilities for sequential recommendation models.

The subpackage is organised so that ``cornac`` can still be imported in an
environment without :mod:`torch`. Only the numpy-only iterators are
re-exported here; the torch-using helpers must be imported explicitly from
their dedicated modules inside ``fit()``::

    # at module top — safe, numpy only
    from ..seq_utils import io_iter, user_io_iter, session_seq_iter, user_seq_iter

    def fit(self, train_set, val_set=None):
        import torch
        from ..seq_utils.losses import get_loss_function
        from ..seq_utils.optim import IndexedAdagradM
        ...

Unified output contract
-----------------------

* **Model forward output**: every sequential model returns an
  ``item_scores`` matrix of shape ``(B, B + N)`` where the diagonal holds the
  positive target scores and the remaining columns are in-batch + sampled
  negatives.
* **Loss functions** (see :mod:`.losses`) all consume that ``(B, B+N)``
  matrix and return a scalar.
* **Iterators** (see :mod:`.iterators`) yield uniform tuples regardless of
  mode:

  =========================  =================================================
  Iterator                   Yielded tuple
  =========================  =================================================
  ``io_iter``                ``(in_uids, in_iids, out_iids, start_mask,
                               valid_id)`` — per-item RNN, session-based.
  ``user_io_iter``           ``(in_uids, in_iids, out_iids, start_mask,
                               valid_id)`` — per-item RNN, session-aware.
  ``session_seq_iter``       ``(in_uids, hist_iids, out_iids)`` — sequence
                               models, session-based.
  ``user_seq_iter``          ``(in_uids, hist_iids, out_iids)`` — sequence
                               models, session-aware.
  =========================  =================================================
"""

from .iterators import (
    io_iter,
    user_io_iter,
    session_seq_iter,
    user_seq_iter,
)

__all__ = [
    "io_iter",
    "user_io_iter",
    "session_seq_iter",
    "user_seq_iter",
]
