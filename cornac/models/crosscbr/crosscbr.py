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

"""Independent tensor implementation of CrossCBR (Ma et al., KDD 2022)."""

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


def graph_tensor(relation, device, rng, dropout=0.0, bipartite=True):
    """Drop edges, then normalize graph degrees or membership row sums."""
    if bipartite:
        left, right = relation.shape
        relation = sp.bmat(
            [
                [sp.csr_matrix((left, left)), relation],
                [relation.T, sp.csr_matrix((right, right))],
            ],
            format="coo",
        )
    else:
        relation = relation.tocoo(copy=True)
    values = relation.data.copy()
    if dropout:
        values *= rng.binomial(1, 1 - dropout, size=len(values))
    graph = sp.csr_matrix((values, (relation.row, relation.col)), shape=relation.shape)
    rows = np.asarray(graph.sum(axis=1)).ravel()
    if bipartite:
        cols = np.asarray(graph.sum(axis=0)).ravel()
        graph = sp.diags(1 / (np.sqrt(rows) + 1e-8)) @ graph
        graph = graph @ sp.diags(1 / (np.sqrt(cols) + 1e-8))
    else:
        graph = sp.diags(1 / (rows + 1e-8)) @ graph
    graph = graph.tocoo()
    indices = torch.as_tensor(np.vstack([graph.row, graph.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(
        indices,
        torch.as_tensor(graph.data, dtype=torch.float32),
        graph.shape,
        device=device,
    ).coalesce()


class Model(nn.Module):
    def __init__(
        self,
        ub,
        ui,
        bi,
        emb_size,
        num_layers,
        temperature,
        aug_type,
        dropouts,
        rng,
        device,
    ):
        super().__init__()
        self.num_users = ub.shape[0]
        self.num_layers = num_layers
        self.temperature = temperature
        self.aug_type = aug_type
        self.dropouts = dropouts
        self.rng = rng
        self.device = device
        self.user_embedding = nn.Parameter(
            torch.empty(ub.shape[0], emb_size, device=device)
        )
        self.bundle_embedding = nn.Parameter(
            torch.empty(ub.shape[1], emb_size, device=device)
        )
        self.item_embedding = nn.Parameter(
            torch.empty(ui.shape[1], emb_size, device=device)
        )
        for embedding in self.parameters():
            nn.init.xavier_normal_(embedding)
        self.relations = (ui, ub, bi)
        self.full_graphs = self._graphs((0.0, 0.0, 0.0))
        self.refresh_graphs()

    def _graphs(self, rates):
        return tuple(
            graph_tensor(relation, self.device, self.rng, rate, bipartite=index < 2)
            for index, (relation, rate) in enumerate(zip(self.relations, rates))
        )

    def refresh_graphs(self):
        self.training_graphs = (
            self._graphs(self.dropouts) if self.aug_type == "ED" else self.full_graphs
        )

    def _propagate(self, graph, targets, rate):
        messages = torch.cat([self.user_embedding, targets], dim=0)
        result = messages
        for layer in range(self.num_layers):
            messages = torch.sparse.mm(graph, messages)
            if self.aug_type == "MD" and self.training:
                messages = F.dropout(messages, p=rate, training=True)
            messages = messages / (layer + 2)
            result = result + F.normalize(messages, dim=1)
        return result[: self.num_users], result[self.num_users :]

    def representations(self):
        ui, ub, bi = self.training_graphs if self.training else self.full_graphs
        item_users, items = self._propagate(ui, self.item_embedding, self.dropouts[0])
        bundle_users, bundles = self._propagate(
            ub, self.bundle_embedding, self.dropouts[1]
        )
        item_bundles = torch.sparse.mm(bi, items)
        if self.aug_type == "MD" and self.training:
            item_bundles = F.dropout(item_bundles, p=self.dropouts[2], training=True)
        return (item_users, bundle_users), (item_bundles, bundles)

    def contrastive_loss(self, item_view, bundle_view):
        logits = F.normalize(item_view, dim=1) @ F.normalize(bundle_view, dim=1).T
        labels = torch.arange(len(logits), device=logits.device)
        return F.cross_entropy(logits / self.temperature, labels)

    def forward(self, users, positives, negatives):
        user_views, bundle_views = self.representations()
        positive_scores = sum(
            (u[users] * b[positives]).sum(dim=1)
            for u, b in zip(user_views, bundle_views)
        )
        negative_scores = sum(
            (u[users] * b[negatives]).sum(dim=1)
            for u, b in zip(user_views, bundle_views)
        )
        bpr = F.softplus(negative_scores - positive_scores).mean()
        contrastive = (
            self.contrastive_loss(user_views[0][users], user_views[1][users])
            + self.contrastive_loss(
                bundle_views[0][positives], bundle_views[1][positives]
            )
        ) / 2
        return bpr, contrastive
