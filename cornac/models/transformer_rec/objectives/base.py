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
"""


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
