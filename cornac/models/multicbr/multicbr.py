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

"""Independent three-view MultiCBR tensor implementation (Ma et al., 2024)."""

import torch
from torch import nn
from torch.nn import functional as F

from ..crosscbr.crosscbr import graph_tensor


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
        ratios,
        view_weights,
        layer_weights,
        rng,
        device,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.temperature = temperature
        self.aug_type = aug_type
        self.ratios = ratios
        self.relations = (ub, ui, bi)
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
        self.register_buffer(
            "view_weights",
            torch.as_tensor(view_weights, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "layer_weights",
            torch.as_tensor(layer_weights, dtype=torch.float32, device=device),
        )
        self.full_graphs = self._graphs((0.0, 0.0, 0.0))
        self.refresh_graphs()

    def _graphs(self, rates):
        ub, ui, bi = self.relations
        return (
            graph_tensor(ub, self.device, self.rng, rates[0]),
            graph_tensor(ui, self.device, self.rng, rates[1]),
            graph_tensor(ui, self.device, self.rng, rates[1], bipartite=False),
            graph_tensor(bi, self.device, self.rng, rates[2]),
            graph_tensor(bi, self.device, self.rng, rates[2], bipartite=False),
        )

    def refresh_graphs(self):
        self.training_graphs = (
            self._graphs(self.ratios) if self.aug_type == "ED" else self.full_graphs
        )

    def _augment(self, features, ratio):
        if not self.training:
            return features
        if self.aug_type == "MD":
            return F.dropout(features, p=ratio, training=True)
        if self.aug_type == "Noise":
            noise = F.normalize(torch.rand_like(features), dim=1)
            return features + ratio * torch.sign(features) * noise
        return features

    def _propagate(self, graph, left, right, view):
        messages = torch.cat([left, right], dim=0)
        result = self.layer_weights[view, 0] * messages
        for depth in range(self.num_layers):
            messages = self._augment(
                torch.sparse.mm(graph, messages), self.ratios[view]
            )
            result = result + self.layer_weights[view, depth + 1] * F.normalize(
                messages, dim=1
            )
        return result[: len(left)], result[len(left) :]

    def representations(self):
        ub, ui, ui_pool, bi, bi_pool = (
            self.training_graphs if self.training else self.full_graphs
        )
        ub_users, ub_bundles = self._propagate(
            ub, self.user_embedding, self.bundle_embedding, 0
        )
        ui_users, ui_items = self._propagate(
            ui, self.user_embedding, self.item_embedding, 1
        )
        ui_bundles = self._augment(torch.sparse.mm(bi_pool, ui_items), self.ratios[2])
        bi_bundles, bi_items = self._propagate(
            bi, self.bundle_embedding, self.item_embedding, 2
        )
        bi_users = self._augment(torch.sparse.mm(ui_pool, bi_items), self.ratios[1])
        users = sum(
            weight * view
            for weight, view in zip(self.view_weights, (ub_users, ui_users, bi_users))
        )
        bundles = sum(
            weight * view
            for weight, view in zip(
                self.view_weights, (ub_bundles, ui_bundles, bi_bundles)
            )
        )
        return users, bundles

    def contrastive_loss(self, representations):
        normalized = F.normalize(representations, dim=1)
        logits = normalized @ normalized.T / self.temperature
        return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def forward(self, users, positives, negatives):
        # The released baseline contrasts one fused pass with itself, not two passes.
        user_vectors, bundle_vectors = self.representations()
        selected_users = user_vectors[users]
        positive = (selected_users * bundle_vectors[positives]).sum(dim=1)
        negative = (selected_users * bundle_vectors[negatives]).sum(dim=1)
        bpr = F.softplus(negative - positive).mean()
        contrastive = (
            self.contrastive_loss(selected_users)
            + self.contrastive_loss(bundle_vectors[positives])
        ) / 2
        return bpr, contrastive
