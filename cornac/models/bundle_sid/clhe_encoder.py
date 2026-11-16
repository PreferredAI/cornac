# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""CF-only CLHE-style item/bundle contrastive encoder, not a CLHE replica.

Only imported during fitting: importing Cornac does not require torch.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..bundle_utils.torch_utils import info_nce
from ...utils import get_rng


class CLHEBundleEncoder(nn.Module):
    """Project items, attend to members, and pool without position embeddings."""

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def item_vectors(self, values, augment=False):
        if augment:
            values = F.dropout(values, p=self.dropout, training=True)
        return self.projection(values)

    def forward(self, values, mask):
        items = self.item_vectors(values)
        attended, _ = self.attention(items, items, items,
                                     key_padding_mask=~mask, need_weights=False)
        hidden = self.norm(items + attended)
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
        return F.normalize(pooled, dim=-1)


def encode_bundles(train_set, item_features, num_epochs=20, batch_size=128,
                   learning_rate=1e-3, temperature=0.2, item_weight=0.1,
                   item_dropout=0.2, max_bundle_items=128, heads=4,
                   seed=None, device="cpu"):
    """Learn two-level contrastive features and return vectors plus diagnostics.

    Bundle views independently drop members while retaining at least one.
    Item views independently mask feature coordinates. Both use in-batch
    negatives. This is a documented CF-only adaptation with no media encoder.
    """
    features = np.asarray(item_features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("item_features must be a finite matrix")
    if features.shape[0] != train_set.num_constituent_items:
        raise ValueError("item features must follow constituent_iid_map")
    if (num_epochs < 1 or batch_size < 2 or temperature <= 0
            or not 0 < item_dropout < 1 or item_weight < 0
            or heads < 1 or features.shape[1] % heads
            or max_bundle_items < 1):
        raise ValueError("invalid CLHE-style encoder settings")
    rng = get_rng(seed)
    torch.manual_seed(0 if seed is None else seed)
    encoder = CLHEBundleEncoder(features.shape[1], heads, item_dropout).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    membership = train_set.bundle_item_matrix.tocsr()
    members, truncated = [], 0
    for bundle in range(membership.shape[0]):
        ids = membership.getrow(bundle).indices.copy()
        if len(ids) > max_bundle_items:
            ids = np.sort(rng.choice(ids, max_bundle_items, replace=False))
            truncated += 1
        members.append(ids)
    eligible = np.array([i for i, ids in enumerate(members) if len(ids)])
    if len(eligible) < 2:
        raise ValueError("contrastive encoding needs at least two nonempty bundles")
    feature_tensor = torch.as_tensor(features, device=device)

    def batch(indices, augment):
        selected = []
        for index in indices:
            ids = members[index]
            if augment:
                keep = rng.random_sample(len(ids)) >= item_dropout
                if not keep.any():
                    keep[rng.randint(len(ids))] = True
                ids = ids[keep]
            selected.append(ids)
        width = max(len(ids) for ids in selected)
        padded = np.zeros((len(selected), width), dtype=np.int64)
        mask = np.zeros(padded.shape, dtype=bool)
        for row, ids in enumerate(selected):
            padded[row, :len(ids)] = ids
            mask[row, :len(ids)] = True
        return feature_tensor[torch.as_tensor(padded, device=device)], torch.as_tensor(mask, device=device)

    losses = []
    for _ in range(num_epochs):
        encoder.train()
        order = rng.permutation(eligible)
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            if len(indices) < 2:
                continue
            a, mask_a = batch(indices, True)
            b, mask_b = batch(indices, True)
            za, zb = encoder(a, mask_a), encoder(b, mask_b)
            bundle_loss = 0.5 * (info_nce(za, zb, temperature) + info_nce(zb, za, temperature))
            item_ids = np.unique(np.concatenate([members[i] for i in indices]))
            if len(item_ids) > 256:
                item_ids = rng.choice(item_ids, 256, replace=False)
            values = feature_tensor[torch.as_tensor(item_ids, device=device)]
            ia, ib = encoder.item_vectors(values, True), encoder.item_vectors(values, True)
            item_loss = 0.5 * (info_nce(ia, ib, temperature) + info_nce(ib, ia, temperature))
            loss = bundle_loss + item_weight * item_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    encoder.eval()
    result = np.zeros((membership.shape[0], features.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(eligible), batch_size):
            indices = eligible[start:start + batch_size]
            values, mask = batch(indices, False)
            result[indices] = encoder(values, mask).cpu().numpy()
    state = {key: value.detach().cpu() for key, value in encoder.state_dict().items()}
    return result, dict(losses=losses, truncated_bundles=truncated,
                        empty_bundles=int(membership.shape[0] - len(eligible)),
                        state_dict=state)
