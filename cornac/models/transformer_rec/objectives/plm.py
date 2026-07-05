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


class PLMObjective(Objective):
    """Permutation Language Modeling (XLNet two-stream), Transformers4Rec-style.

    Recipe followed
    ---------------
    Adapted from NVIDIA Merlin Transformers4Rec,
    ``transformers4rec/torch/masking.py`` class
    :class:`PermutationLanguageModeling` (method
    ``_compute_masked_targets_extended``, training branch). The ``perm_mask``
    construction is reproduced verbatim: a random factorization order is drawn
    per row, non-target positions are pinned to ``-1`` (so every query may
    attend to them and they never leak into a masked target), and

        ``perm_mask[b, q, k] = (perm_index[q] <= perm_index[k]) & mask_labels[k]``

    which forbids query ``q`` from attending to a masked key ``k`` unless ``k``
    precedes ``q`` in the factorization order. Because ``perm_index[q] ==
    perm_index[k]`` for ``q == k``, a target can never attend to itself, so the
    original item ids are fed as inputs (no ``mask_idx`` replacement during
    training) and XLNet's query (``g``) stream produces each prediction.

    Deviations from Transformers4Rec
    --------------------------------
    * Target selection uses an i.i.d. Bernoulli(``mask_prob``) draw over the
      non-pad positions (as in MLM), not T4R's span-based sampling. This keeps
      the objective consistent with the other TransformerRec objectives and is
      the behaviour requested for this integration.
    * ``target_mapping`` is built in the compact ``(B, K, T)`` form (one row per
      selected target, ``K`` = per-batch max target count, short rows zero-
      padded and tracked by a validity mask) so XLNet returns ``(B, K, D)``
      directly. T4R instead uses a full ``(B, T, T)`` identity and relies on the
      padded labels to drop non-targets; the compact form is equivalent but
      avoids scoring the non-target positions.

    Parameters
    ----------
    pad_idx: int
        Padding item index (``item_num``).
    mask_idx: int
        Mask item index (``item_num + 1``), used only at inference time.
    rng: numpy.random.RandomState
        Random state used for target selection and factorization order.
    mask_prob: float, optional, default: 0.2
        Per-position probability of selecting a non-pad item as a prediction
        target. At least one target is kept per row.
    """

    VALID_ATTENTION = ("bidirectional",)
    uses_mask_token = True

    def __init__(self, pad_idx, mask_idx, rng, mask_prob=0.2):
        super().__init__(pad_idx, mask_idx, rng)
        self.mask_prob = mask_prob

    def _select_targets(self, non_pad):
        """Bernoulli target selection over non-pad positions.

        Guarantees at least one target and at least one visible (non-target)
        item per row, following the T4R MLM safeguards.
        """
        B, T = non_pad.shape
        mask_labels = (self.rng.random((B, T)) < self.mask_prob) & non_pad
        for b in range(B):
            valid = np.where(non_pad[b])[0]
            if len(valid) == 0:
                continue
            if not mask_labels[b].any():
                mask_labels[b, self.rng.choice(valid)] = True
            # If every non-pad item is a target, unmask one to keep context.
            if len(valid) > 1 and mask_labels[b].sum() == len(valid):
                masked = np.where(mask_labels[b])[0]
                mask_labels[b, self.rng.choice(masked)] = False
        return mask_labels

    def compute_loss(self, model, seqs, sample_negatives, loss_fn, loss_kwargs):
        device = model.dev
        B, T = seqs.shape
        non_pad = seqs != self.pad_idx
        mask_labels = self._select_targets(non_pad)

        # Random factorization-order attention mask (T4R recipe).
        perm_mask = np.zeros((B, T, T), dtype=np.float32)
        for b in range(B):
            perm_index = self.rng.permutation(T).astype(np.int64)
            perm_index[~mask_labels[b]] = -1
            perm_mask[b] = (
                perm_index[:, None] <= perm_index[None, :]
            ) & mask_labels[b][None, :]

        # Compact target mapping (B, K, T) with a validity mask.
        counts = mask_labels.sum(axis=1)
        K = int(counts.max())
        target_mapping = np.zeros((B, K, T), dtype=np.float32)
        target_items = np.zeros((B, K), dtype=np.int64)
        validity = np.zeros((B, K), dtype=bool)
        for b in range(B):
            positions = np.where(mask_labels[b])[0]
            for j, p in enumerate(positions):
                target_mapping[b, j, p] = 1.0
                target_items[b, j] = seqs[b, p]
                validity[b, j] = True

        inputs = torch.as_tensor(seqs, dtype=torch.long, device=device)
        perm_t = torch.as_tensor(perm_mask, device=device)
        tm_t = torch.as_tensor(target_mapping, device=device)

        # g-stream outputs, one per target slot: (B, K, D).
        g = model.encode(inputs, perm_mask=perm_t, target_mapping=tm_t)

        valid_t = torch.as_tensor(validity, device=device)
        hidden = g[valid_t]  # (M, D)
        targets = torch.as_tensor(target_items, dtype=torch.long, device=device)[
            valid_t
        ]  # (M,)
        M = targets.shape[0]

        out_iids = targets
        n_neg = loss_kwargs.get("n_sample", 0) or 0
        if sample_negatives is not None and n_neg > 0:
            negs = torch.as_tensor(
                sample_negatives(n_neg), dtype=torch.long, device=device
            )
            out_iids = torch.cat([targets, negs])

        scores = model.score_positions(hidden, out_iids)  # (M, M+N)
        return loss_fn(scores, out_iids=out_iids, batch_size=M, **loss_kwargs)

    def prepare_score_input(self, history, max_len, pad_idx):
        """Mask-append serving (plain bidirectional pass, as T4R does).

        Truncate to the last ``max_len - 1`` items, append ``mask_idx``, then
        left-pad to ``max_len``. The final (mask) position is read.
        """
        hist = list(history)[-(max_len - 1):]
        seq = hist + [self.mask_idx]
        seq = [pad_idx] * (max_len - len(seq)) + seq
        return seq, -1
