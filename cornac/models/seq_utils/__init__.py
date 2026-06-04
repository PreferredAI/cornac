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
"""Shared utilities for session-based sequential recommendation models.

* **Model forward output**: every sequential model returns an
  ``item_scores`` matrix of shape ``(B, B + N)`` where the diagonal holds
  the positive target scores and the remaining columns are in-batch +
  sampled negatives.
* **Loss functions** (see :mod:`.losses`) all consume that ``(B, B+N)``
  matrix and return a scalar.
* **Iterators** (see :mod:`.iterators`) yield uniform tuples:

  =========================  =================================================
  Iterator                   Yielded tuple
  =========================  =================================================
  ``io_iter``                ``(in_iids, out_iids, start_mask, valid_id)``
                               -- per-item RNN, session-based.
  ``session_seq_iter``       ``(in_uids, hist_iids, out_iids)`` -- sequence
                               models, session-based.
  =========================  =================================================
"""

from .iterators import io_iter, session_seq_iter
from .selection import val_score

__all__ = [
    "io_iter",
    "session_seq_iter",
    "val_score",
]
