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

import numpy as np
import torch

from .base import Objective
from .clm import _build_out_iids


class MLMObjective(Objective):
    """Masked language modeling (BERT4Rec-style Cloze objective).

    Random non-pad positions are replaced by ``mask_idx`` and the model must
    recover the original items. Masking always replaces with the mask token
    (no 80/10/10 split). Requires a bidirectional backbone.
    """

    VALID_ATTENTION = ("bidirectional",)
    uses_mask_token = True

    def __init__(self, pad_idx, mask_idx, rng, mask_prob=0.2):
        super().__init__(pad_idx, mask_idx, rng)
        self.mask_prob = mask_prob

    def compute_loss(self, model, seqs, sample_negatives, loss_fn, loss_kwargs):
        seqs_t = torch.as_tensor(seqs, dtype=torch.long, device=model.dev)
        seqs_np = np.asarray(seqs)
        B, T = seqs_np.shape

        # Sample mask positions among NON-pad positions; force >= 1 per row.
        mask_matrix = np.zeros((B, T), dtype=bool)
        for b in range(B):
            positions = np.nonzero(seqs_np[b] != self.pad_idx)[0]
            if len(positions) == 0:
                continue
            chosen = positions[self.rng.rand(len(positions)) < self.mask_prob]
            if len(chosen) == 0:
                chosen = self.rng.choice(positions, size=1)
            mask_matrix[b, chosen] = True

        mask_t = torch.as_tensor(mask_matrix, device=model.dev)
        inputs = seqs_t.clone()
        inputs[mask_t] = self.mask_idx  # always replace (no 80/10/10)

        hidden = model.encode(inputs)  # (B, T, D)
        hidden_flat = hidden[mask_t]  # (M, D)
        target_flat = seqs_t[mask_t]  # (M,)

        out_iids = _build_out_iids(
            target_flat, sample_negatives, loss_kwargs, model.dev
        )
        scores = model.score_positions(hidden_flat, out_iids)
        return loss_fn(
            scores,
            out_iids=out_iids,
            batch_size=hidden_flat.size(0),
            **loss_kwargs,
        )

    def prepare_score_input(self, history, max_len, pad_idx):
        """Append a mask token after the history and read it.

        Keep the last ``max_len - 1`` items, append ``mask_idx``, and
        left-pad to ``max_len``; the mask position (last) is scored.
        """
        hist = list(history)[-(max_len - 1):]
        input_iids = hist + [self.mask_idx]
        input_iids = [pad_idx] * (max_len - len(input_iids)) + input_iids
        return input_iids, -1
