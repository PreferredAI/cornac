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
import torch.nn as nn
import torch.nn.functional as F

from .base import Objective
from .clm import _build_out_iids


class RTDObjective(Objective):
    """Replacement Token Detection (ELECTRA), tied-generator variant.

    Recipe followed
    ---------------
    Adapted from NVIDIA Merlin Transformers4Rec,
    ``transformers4rec/torch/masking.py`` class
    :class:`ReplacementLanguageModeling` (which subclasses
    :class:`MaskedLanguageModeling`) with its ``get_fake_tokens`` /
    ``sample_from_softmax`` corruption step, in the **tied-generator**
    configuration (T4R's ``rtd_tied_generator``): the main transformer body
    plays both the generator and the discriminator roles. Per batch:

    1. MLM-mask the batch (Bernoulli ``mask_prob`` over non-pad positions, at
       least one masked and at least one visible per row).
    2. Generator pass: the main model encodes the masked sequence; masked-
       position hidden states are scored against the shared ``item_emb`` under
       the same ``(M, M+N)`` diagonal-positive contract used by every other
       objective, giving the MLM (generator) loss.
    3. Replacements are sampled from the model's own softmax over the real
       items (``torch.multinomial``, detached so no gradient flows through
       sampling) and scattered into the masked positions to build a corrupted
       sequence (T4R ``get_fake_tokens``). Positions whose sample equals the
       original count as "original" (ELECTRA convention).
    4. Discriminator pass: the same body encodes the corrupted sequence; a
       per-position ``Linear(D, 1)`` head classifies original-vs-replaced over
       every non-pad position with BCE-with-logits.
    5. ``total = mlm_loss + rtd_lambda * disc_loss``.

    Serving head
    ------------
    Standard MLM serving: append ``mask_idx`` to the history and read its
    position through the main body (inherited default ``predict_scores``).
    Because the same body is trained on masked inputs in step (2), the
    serving path is fully trained — and independent of ``disc_head``, so
    ``model_selection='best'`` snapshots of the main model alone cover it.

    Deviations from ELECTRA / Transformers4Rec
    ------------------------------------------
    * ELECTRA's separate *small* generator (T4R's untied variant with
      ``generator_size_ratio``) is not implemented: with zero-shot serving
      there is no fine-tuning step to exploit a discriminator-only body, so
      the untrained-serving-head problem makes the untied variant unusable
      here (empirically test AUC < 0.5).
    * Replacements are sampled with plain ``torch.multinomial`` on the softmax
      over the real items only (pad/mask columns excluded), rather than T4R's
      Gumbel-argmax; both are unbiased categorical draws.

    Parameters
    ----------
    pad_idx: int
        Padding item index (``item_num``).
    mask_idx: int
        Mask item index (``item_num + 1``).
    rng: numpy.random.RandomState
        Random state for masked-position selection.
    mask_prob: float, optional, default: 0.2
        Per-position masking probability. At least one item is masked per row.
    rtd_lambda: float, optional, default: 1.0
        Weight of the discriminator (RTD) loss relative to the MLM loss.
        ``rtd_lambda -> 0`` recovers plain MLM. ELECTRA's original 50 is tuned
        for fine-tuning pipelines; for zero-shot ranking the discriminator
        term competes with the item-embedding alignment, and Diginetica
        sweeps show ranking quality degrading monotonically with lambda —
        tune downward if ranking metrics matter most.
    """

    VALID_ATTENTION = ("bidirectional",)
    uses_mask_token = True

    def __init__(self, pad_idx, mask_idx, rng, mask_prob=0.2, rtd_lambda=1.0):
        super().__init__(pad_idx, mask_idx, rng)
        self.mask_prob = mask_prob
        self.rtd_lambda = rtd_lambda
        self.disc_head = None

    def build(self, model, device):
        self.disc_head = nn.Linear(model.item_emb.embedding_dim, 1)
        self.disc_head.to(device)

    def parameters(self):
        return list(self.disc_head.parameters())

    def _mlm_mask(self, non_pad):
        """Bernoulli MLM masking with T4R's at-least-one / not-all safeguards."""
        B, T = non_pad.shape
        mask_labels = (self.rng.random((B, T)) < self.mask_prob) & non_pad
        for b in range(B):
            valid = np.where(non_pad[b])[0]
            if len(valid) == 0:
                continue
            if not mask_labels[b].any():
                mask_labels[b, self.rng.choice(valid)] = True
            if len(valid) > 1 and mask_labels[b].sum() == len(valid):
                masked = np.where(mask_labels[b])[0]
                mask_labels[b, self.rng.choice(masked)] = False
        return mask_labels

    def compute_loss(self, model, seqs, sample_negatives, loss_fn, loss_kwargs):
        device = model.dev
        non_pad = seqs != self.pad_idx
        mask_labels = self._mlm_mask(non_pad)

        seqs_t = torch.as_tensor(seqs, dtype=torch.long, device=device)
        non_pad_t = torch.as_tensor(non_pad, device=device)
        mask_t = torch.as_tensor(mask_labels, device=device)  # (B, T) bool

        # (1-2) Generator pass: MLM through the main body.
        masked_ids = seqs_t.clone()
        masked_ids[mask_t] = self.mask_idx
        hidden = model.encode(masked_ids)  # (B, T, D)
        mlm_hidden = hidden[mask_t]  # (M, D)
        targets = seqs_t[mask_t]  # (M,)

        out_iids = _build_out_iids(targets, sample_negatives, loss_kwargs, device)
        mlm_scores = model.score_positions(mlm_hidden, out_iids)  # (M, M+N)
        mlm_loss = loss_fn(
            mlm_scores, out_iids=out_iids, batch_size=targets.shape[0], **loss_kwargs
        )

        # (3) Sample replacements from the model's own predictions (detached).
        with torch.no_grad():
            real_ids = torch.arange(model.item_num, device=device)
            real_scores = model.score_positions(mlm_hidden, real_ids)  # (M, item_num)
            probs = torch.softmax(real_scores, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (M,)

        corrupted = seqs_t.clone()
        corrupted[mask_t] = sampled  # same row-major order as mlm_hidden / targets

        # (4) Discriminator pass: same body classifies original vs replaced.
        disc_hidden = model.encode(corrupted)  # (B, T, D)
        disc_logits = self.disc_head(disc_hidden).squeeze(-1)  # (B, T)
        disc_labels = (corrupted != seqs_t).float()  # 1 where replaced
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_pad_t], disc_labels[non_pad_t]
        )

        # (5) Joint objective.
        return mlm_loss + self.rtd_lambda * disc_loss

    def prepare_score_input(self, history, max_len, pad_idx):
        """Mask-append serving through the main body (trained by the MLM pass)."""
        hist = list(history)[-(max_len - 1):]
        seq = hist + [self.mask_idx]
        seq = [pad_idx] * (max_len - len(seq)) + seq
        return seq, -1
