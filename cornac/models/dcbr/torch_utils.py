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
"""Graph propagation and losses for DCBR.

This module imports torch at the top; import it only from inside ``fit``.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def sp_to_torch(matrix, device=None):
    """SciPy sparse matrix -> coalesced float32 torch sparse COO tensor."""
    coo = sp.coo_matrix(matrix)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    tensor = torch.sparse_coo_tensor(
        indices, values, coo.shape, check_invariants=False
    ).coalesce()
    return tensor if device is None else tensor.to(device)


def perturb(h, eps):
    """Noise augmentation h + sign(h) * eps * unit-norm uniform noise (SimGCL style)."""
    if eps <= 0:
        return h
    noise = F.normalize(torch.rand_like(h), dim=-1)
    return h + torch.sign(h) * noise * eps


def lightgcn_propagate(
    graph, x, num_layers, layer_weights=None, normalize=False, noise_eps=0.0
):
    """LightGCN propagation without feature transforms or nonlinearities.

    Parameters
    ----------
    graph: torch sparse tensor
        Normalized (n x n) adjacency.
    x: torch.Tensor
        Layer-0 embeddings (n x d).
    num_layers: int
        Number of propagation steps.
    layer_weights: sequence of float, optional
        Weights of layers 0..num_layers. Default: plain sum.
    normalize: bool
        L2-normalize each propagated layer before combining (layer 0 is kept raw),
        as in CrossCBR and MultiCBR.
    noise_eps: float
        Noise magnitude added after every propagation step (MultiCBR).

    Returns
    -------
    torch.Tensor of shape (n x d)
    """
    layers = [x]
    h = x
    for _ in range(num_layers):
        h = torch.sparse.mm(graph, h)
        h = perturb(h, noise_eps)
        layers.append(F.normalize(h, p=2, dim=1) if normalize else h)
    if layer_weights is None:
        return torch.stack(layers, dim=0).sum(dim=0)
    if len(layer_weights) != num_layers + 1:
        raise ValueError("layer_weights must have num_layers + 1 values")
    return sum(w * layer for w, layer in zip(layer_weights, layers))


def bpr_loss(pos_scores, neg_scores):
    """Mean BPR loss -log sigmoid(pos - neg) = softplus(neg - pos)."""
    return F.softplus(neg_scores - pos_scores).mean()


def info_nce(a, b, temperature):
    """InfoNCE over a batch: row i of ``a`` is positive with row i of ``b``.

    Cosine similarities divided by ``temperature``; every row of ``b``
    (including the positive) is in the denominator. Returns the mean loss.
    """
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    logits = a @ b.T / temperature
    targets = torch.arange(a.shape[0], device=a.device)
    return F.cross_entropy(logits, targets)
