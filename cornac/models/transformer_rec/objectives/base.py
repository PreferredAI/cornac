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
"""Training-objective contract for :class:`TransformerRecModel`.

An :class:`Objective` turns left-padded whole sessions into inputs, loss
positions, and targets; runs the model; and returns a scalar loss computed
by one of the shared :mod:`cornac.models.seq_utils` loss functions against a
``(M, M + N)`` score matrix.

Shared helpers used by several objectives live here too:
:func:`bernoulli_mask` (MLM / PLM / RTD position selection) and
:func:`build_out_iids` (positives + sampled negatives concatenation).
"""

import numpy as np
import torch


def bernoulli_mask(non_pad, mask_prob, rng):
    """Bernoulli position selection with the T4R MLM safeguards.

    Draws i.i.d. Bernoulli(``mask_prob``) over the non-pad positions, then
    guarantees per row at least one selected position and, when the row has
    more than one non-pad item, at least one visible (unselected) item.

    Parameters
    ----------
    non_pad : numpy.ndarray of bool, shape (B, T)
        True where the position holds a real item.
    mask_prob : float
        Per-position selection probability.
    rng : numpy.random.RandomState

    Returns
    -------
    numpy.ndarray of bool, shape (B, T)
        True at the selected positions.
    """
    B, T = non_pad.shape
    mask_labels = (rng.random((B, T)) < mask_prob) & non_pad
    for b in range(B):
        valid = np.where(non_pad[b])[0]
        if len(valid) == 0:
            continue
        if not mask_labels[b].any():
            mask_labels[b, rng.choice(valid)] = True
        # If every non-pad item is selected, unselect one to keep context.
        if len(valid) > 1 and mask_labels[b].sum() == len(valid):
            masked = np.where(mask_labels[b])[0]
            mask_labels[b, rng.choice(masked)] = False
    return mask_labels


def build_out_iids(target_flat, sample_negatives, loss_kwargs, device):
    """Concatenate positives with sampled negatives.

    The negative count is read from ``loss_kwargs['n_sample']`` (mirroring
    the family's ``loss_kwargs = dict(..., n_sample=...)`` convention);
    absent or zero yields in-batch negatives only.
    """
    n_sample = loss_kwargs.get("n_sample", 0)
    if sample_negatives is None or not n_sample:
        return target_flat
    negatives = torch.as_tensor(
        sample_negatives(n_sample), dtype=torch.long, device=device
    )
    return torch.cat([target_flat, negatives])


class Objective:
    """Base training objective (next-item / causal defaults)."""

    #: Backbone attention types this objective is compatible with.
    VALID_ATTENTION = ("causal", "bidirectional")
    #: Whether the objective feeds the ``mask_idx`` token to the model.
    uses_mask_token = False

    def __init__(self, pad_idx, mask_idx, rng):
        self.pad_idx, self.mask_idx, self.rng = pad_idx, mask_idx, rng

    def build(self, model, device):
        """One-time hook after :class:`TransformerRecModel` construction.

        Objectives that need extra sub-modules (e.g. RTD builds its generator
        here) create them in this hook. Default: no-op.
        """

    def parameters(self):
        """Extra trainable parameters beyond the model's. Default: ``[]``."""
        return []

    def compute_loss(self, model, seqs, sample_negatives, loss_fn, loss_kwargs):
        """Compute the training loss for one batch of sessions.

        Parameters
        ----------
        model : TransformerRecModel
        seqs : numpy.ndarray, shape (B, T)
            Left-padded whole sessions.
        sample_negatives : callable or None
            ``n -> numpy.ndarray`` returning ``n`` sampled negative item ids.
        loss_fn : callable
            A loss from :mod:`cornac.models.seq_utils.losses`.
        loss_kwargs : dict
            Extra keyword arguments forwarded to ``loss_fn``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        raise NotImplementedError

    def prepare_score_input(self, history, max_len, pad_idx):
        """Turn a known history into a padded model input for inference.

        Parameters
        ----------
        history : list of int
            Known item ids (non-empty).
        max_len : int
        pad_idx : int

        Returns
        -------
        (list of int, int)
            ``(input_iids, read_position)`` where ``input_iids`` has length
            exactly ``max_len`` (left-padded). Default (causal / next-item):
            keep the last ``max_len`` items and read the last position.
        """
        hist = list(history)[-max_len:]
        input_iids = [pad_idx] * (max_len - len(hist)) + hist
        return input_iids, -1

    def predict_scores(self, model, input_iids, read_position):
        """Inference scoring hook. Default: dot-product head via ``predict``."""
        return model.predict(None, input_iids, read_position)
