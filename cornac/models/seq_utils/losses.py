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
"""Loss functions for sequential recommendation models.

Importing this module requires :mod:`torch`. To respect Cornac's optional-
dependency convention, only import it from inside a model's ``fit()`` method
(or another scope guaranteed to be reached only when torch is available).
"""

import torch
import torch.nn.functional as F


def softmax_neg(X):
    """Softmax over negatives, masking out the diagonal (positives)."""
    hm = 1.0 - torch.eye(*X.shape, out=torch.empty_like(X))
    X = X * hm
    e_x = torch.exp(X - X.max(dim=1, keepdim=True)[0]) * hm
    if e_x.size(0) == 1:
        return e_x
    return e_x / (e_x.sum(dim=1, keepdim=True) + 1e-24)


def bpr_loss(item_scores, **kwargs):
    """BPR pairwise logsigmoid loss against in-batch negatives.

    Parameters
    ----------
    item_scores: torch.Tensor of shape (B, B+N)
        Score matrix with positives on the diagonal.
    """
    pos = torch.diag(item_scores)
    pos = pos.reshape(pos.shape[0], -1)
    logits = F.logsigmoid(pos - item_scores)
    mask = 1.0 - torch.eye(*logits.shape, out=torch.empty_like(logits))
    loss = -torch.sum(logits * mask)
    return loss / logits.size(0) / max(logits.size(1) - 1, 1)


def top1_loss(item_scores, n_sample=0, **kwargs):
    """TOP1 ranking loss from Hidasi et al. (2015)."""
    target = torch.diag(item_scores)
    target = target.reshape(target.shape[0], -1)
    return torch.sum(
        torch.mean(
            torch.sigmoid(item_scores - target) + torch.sigmoid(item_scores**2),
            dim=1,
        )
        - torch.sigmoid(target**2) / (item_scores.size(0) + n_sample)
    ) / item_scores.size(0)


def xe_softmax_loss(item_scores, out_iids=None, P0=None, logq=0.0, sample_alpha=0.5, batch_size=None, **kwargs):
    """Cross-entropy with softmax over in-batch + sampled negatives.

    Supports an optional logQ correction (Hidasi & Karatzoglou, 2018) when
    ``P0`` (item popularity prior) and ``logq > 0`` are provided.
    """
    if logq > 0 and P0 is not None and out_iids is not None and batch_size is not None:
        item_scores = item_scores - logq * torch.log(
            torch.cat([P0[out_iids[:batch_size]], P0[out_iids[batch_size:]] ** sample_alpha])
        )
    X = torch.exp(item_scores - item_scores.max(dim=1, keepdim=True)[0])
    X = X / (X.sum(dim=1, keepdim=True) + 1e-24)
    return -torch.sum(torch.log(torch.diag(X) + 1e-24)) / item_scores.size(0)


def bpr_max_loss(item_scores, bpreg=1.0, elu_param=0.5, **kwargs):
    """BPR-max with softmax-weighted negatives and L2 regularisation on scores."""
    if elu_param > 0:
        item_scores = F.elu(item_scores, elu_param)
    softmax_scores = softmax_neg(item_scores)
    target = torch.diag(item_scores)
    target = target.reshape(target.shape[0], -1)
    return torch.sum(
        -torch.log(torch.sum(torch.sigmoid(target - item_scores) * softmax_scores, dim=1) + 1e-24)
        + bpreg * torch.sum((item_scores**2) * softmax_scores, dim=1)
    ) / item_scores.size(0)


def bce_loss(item_scores, **kwargs):
    """Binary cross-entropy treating the diagonal as positive and all other
    columns (in-batch negatives + sampled negatives) as negatives.
    """
    B, N = item_scores.shape
    targets = torch.zeros_like(item_scores)
    targets[torch.arange(B), torch.arange(B)] = 1.0
    return F.binary_cross_entropy_with_logits(item_scores, targets)


def ce_loss(item_scores, **kwargs):
    """Standard cross-entropy where the target class is the in-batch diagonal."""
    targets = torch.arange(item_scores.size(0), device=item_scores.device, dtype=torch.long)
    return F.cross_entropy(item_scores, targets)


LOSS_FUNCTIONS = {
    "bpr": bpr_loss,
    "top1": top1_loss,
    "cross-entropy": xe_softmax_loss,
    "xe_softmax": xe_softmax_loss,
    "softmax": xe_softmax_loss,
    "bpr-max": bpr_max_loss,
    "bce": bce_loss,
    "ce": ce_loss,
}


def get_loss_function(name):
    """Look up a loss function by name. Raises ``ValueError`` if unknown."""
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss '{name}'. Supported: {sorted(set(LOSS_FUNCTIONS))}")
    return LOSS_FUNCTIONS[name]
