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

import torch

from .base import Objective, build_out_iids


class CLMObjective(Objective):
    """Causal language modeling: predict every next item from its prefix.

    Requires a causal backbone in the whole-session path. The legacy
    prefix path (:meth:`compute_loss_prefix`) reproduces the original
    BERT4Rec / GPT2Rec last-position training and is valid for any attention
    type (the recommender enforces that).
    """

    VALID_ATTENTION = ("causal",)
    uses_mask_token = False

    def __init__(self, pad_idx, mask_idx, rng):
        super().__init__(pad_idx, mask_idx, rng)

    def compute_loss(self, model, seqs, sample_negatives, loss_fn, loss_kwargs):
        seqs_t = torch.as_tensor(seqs, dtype=torch.long, device=model.dev)
        inputs = seqs_t[:, :-1]
        targets = seqs_t[:, 1:]

        # Valid loss positions are where the INPUT token is real. With left
        # padding this also guarantees a real target and never asks the model
        # to predict the first item from an empty prefix.
        valid = inputs != self.pad_idx  # (B, T-1)

        hidden = model.encode(inputs)  # (B, T-1, D)
        hidden_flat = hidden[valid]  # (M, D)
        target_flat = targets[valid]  # (M,)

        out_iids = build_out_iids(
            target_flat, sample_negatives, loss_kwargs, model.dev
        )
        scores = model.score_positions(hidden_flat, out_iids)
        return loss_fn(
            scores,
            out_iids=out_iids,
            batch_size=hidden_flat.size(0),
            **loss_kwargs,
        )

    def compute_loss_prefix(self, model, hist_iids, out_iids, loss_fn, loss_kwargs):
        """Legacy last-position prefix path (BERT4Rec / GPT2Rec training).

        Parameters
        ----------
        model : TransformerRecModel
        hist_iids : torch.LongTensor, shape (B, T)
            Left-padded prefixes from ``session_seq_iter``.
        out_iids : torch.LongTensor, shape (B + N,)
            In-batch positives followed by shared negatives.
        """
        hidden = model.encode(hist_iids)[:, -1, :]
        scores = model.score_positions(hidden, out_iids)
        return loss_fn(
            scores,
            out_iids=out_iids,
            batch_size=hist_iids.size(0),
            **loss_kwargs,
        )
