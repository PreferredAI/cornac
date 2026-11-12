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

"""Hierarchical strategy-aware graph transformer for catalog bundles."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..crosscbr.crosscbr import graph_tensor


def neighbor_tokens(relation, maximum, device):
    """Use ascending mapped neighbor IDs with optional truncation and padding."""
    relation = relation.tocsr(copy=True)
    relation.sort_indices()
    width = max(1, int(np.diff(relation.indptr).max()))
    if maximum > 0:
        width = min(width, maximum)
    tokens = np.zeros((relation.shape[0], width), dtype=np.int64)
    mask = np.ones_like(tokens, dtype=bool)
    for row in range(relation.shape[0]):
        neighbors = relation.indices[relation.indptr[row] : relation.indptr[row + 1]][
            :width
        ]
        tokens[row, : len(neighbors)] = neighbors
        mask[row, : len(neighbors)] = False
    return (
        torch.as_tensor(tokens, device=device),
        torch.as_tensor(mask, device=device),
    )


class LightAttention(nn.Module):
    """Single-head Q/K attention with identity values and masked mean readout."""

    def __init__(self, emb_size, dropout, device):
        super().__init__()
        self.query = nn.Linear(emb_size, emb_size, bias=False).to(device)
        self.key = nn.Linear(emb_size, emb_size, bias=False).to(device)
        self.dropout = dropout

    def forward(self, features, tokens, mask, prior, alpha, batch_size):
        pooled = []
        for start in range(0, len(tokens), batch_size):
            indices = tokens[start : start + batch_size]
            padding = mask[start : start + batch_size]
            values = features[indices]
            logits = self.query(values) @ self.key(values).transpose(-2, -1)
            logits = logits / (features.shape[1] ** 0.5)
            attention = logits.masked_fill(padding[:, None, :], -1e9).softmax(dim=-1)
            outputs = F.dropout(
                attention @ values, p=self.dropout, training=self.training
            )
            outputs = outputs.masked_fill(padding[:, :, None], 0)
            counts = (~padding).sum(dim=1, keepdim=True)
            pooled.append(outputs.sum(dim=1) / (counts + 1e-9))
        return alpha * torch.cat(pooled) + (1 - alpha) * prior


class Model(nn.Module):
    def __init__(
        self,
        ub,
        ui,
        bi,
        emb_size,
        num_ui_layers,
        num_trans_layers,
        max_bundle_items,
        max_user_bundles,
        gcn_norm,
        ub_alpha,
        bi_alpha,
        dropout,
        batch_size,
        rng,
        device,
    ):
        super().__init__()
        self.num_ui_layers = num_ui_layers
        self.num_trans_layers = num_trans_layers
        self.gcn_norm = gcn_norm
        self.ub_alpha = ub_alpha
        self.bi_alpha = bi_alpha
        self.batch_size = batch_size
        self.user_embedding = nn.Parameter(
            torch.empty(ub.shape[0], emb_size, device=device)
        )
        self.item_embedding = nn.Parameter(
            torch.empty(ui.shape[1], emb_size, device=device)
        )
        self.bundle_embedding = nn.Parameter(
            torch.empty(ub.shape[1], emb_size, device=device)
        )
        for embedding in self.parameters():
            nn.init.xavier_uniform_(embedding)
        self.bi_attention = nn.ModuleList(
            [LightAttention(emb_size, dropout, device) for _ in range(num_trans_layers)]
        )
        self.ub_attention = nn.ModuleList(
            [LightAttention(emb_size, dropout, device) for _ in range(num_trans_layers)]
        )
        self.ui_graph = graph_tensor(ui, device, rng)
        self.ub_graph = graph_tensor(ub, device, rng)
        self.bi_graph = graph_tensor(bi, device, rng)
        tokens, mask = neighbor_tokens(bi, max_bundle_items, device)
        self.register_buffer("bi_tokens", tokens)
        self.register_buffer("bi_mask", mask)
        tokens, mask = neighbor_tokens(ub, max_user_bundles, device)
        self.register_buffer("ub_tokens", tokens)
        self.register_buffer("ub_mask", mask)

    @staticmethod
    def _propagate(graph, left, right):
        propagated = torch.sparse.mm(graph, torch.cat([left, right], dim=0))
        return propagated[: len(left)], propagated[len(left) :]

    def _content(self):
        messages = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        result = messages
        for depth in range(self.num_ui_layers):
            messages = torch.sparse.mm(self.ui_graph, messages)
            if self.gcn_norm:
                messages = messages / (depth + 1)
                result = result + F.normalize(messages, dim=1)
            else:
                result = result + messages
        return result[: len(self.user_embedding)], result[len(self.user_embedding) :]

    def representations(self):
        users, items, bundles = (
            self.user_embedding,
            self.item_embedding,
            self.bundle_embedding,
        )
        content_users, content_items = self._content()
        for layer in range(self.num_trans_layers):
            user_prior, bundle_prior = self._propagate(self.ub_graph, users, bundles)
            bundle_strategy, item_strategy = self._propagate(
                self.bi_graph, bundles, items
            )
            items = content_items + item_strategy
            bundles = (
                self.bi_attention[layer](
                    items,
                    self.bi_tokens,
                    self.bi_mask,
                    bundle_strategy,
                    self.bi_alpha,
                    self.batch_size,
                )
                + bundle_prior
            )
            users = (
                self.ub_attention[layer](
                    bundles,
                    self.ub_tokens,
                    self.ub_mask,
                    user_prior,
                    self.ub_alpha,
                    self.batch_size,
                )
                + content_users
            )
        return users, items, bundles

    def regularization(self):
        return (
            self.user_embedding.square().sum()
            + self.item_embedding.square().sum()
            + self.bundle_embedding.square().sum()
        )

    def forward(self, users, positives, negatives):
        user_vectors, _, bundle_vectors = self.representations()
        positive = (user_vectors[users] * bundle_vectors[positives]).sum(dim=1)
        negative = (user_vectors[users] * bundle_vectors[negatives]).sum(dim=1)
        return F.softplus(negative - positive).mean(), self.regularization()
