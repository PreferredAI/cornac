# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Collaborative constituent features and order-invariant bundle aggregation."""

import numpy as np

from ..bundle_utils.graph import binary, row_norm, user_item_matrix
from ...utils import get_rng


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def mean_bundle_features(train_set, item_features):
    """Pool on the constituent item axis, not Cornac's bundle target axis."""
    features = np.asarray(item_features, dtype=np.float32)
    if (features.ndim != 2
            or features.shape[0] != train_set.num_constituent_items
            or features.shape[1] == 0 or not np.isfinite(features).all()):
        raise ValueError("item_features must be finite [num_constituent_items, dim]")
    return normalize_rows(row_norm(binary(train_set.bundle_item_matrix)) @ features)


def _negative_items(matrix, users, rng):
    """Uniform complement sampling with an exact fallback after rejection."""
    negatives = rng.randint(matrix.shape[1], size=len(users))
    for position, user in enumerate(users):
        observed = matrix.indices[matrix.indptr[user]:matrix.indptr[user + 1]]
        if len(observed) == matrix.shape[1]:
            raise ValueError("cannot sample a negative for a saturated user")
        for _ in range(32):
            if negatives[position] not in observed:
                break
            negatives[position] = rng.randint(matrix.shape[1])
        else:
            candidates = np.setdiff1d(np.arange(matrix.shape[1]), observed)
            negatives[position] = rng.choice(candidates)
    return negatives


def collaborative_item_features(train_set, emb_size=64, num_layers=1,
                                num_epochs=10, batch_size=2048,
                                learning_rate=1e-3, lambda_reg=1e-4,
                                seed=None, device="cpu"):
    """Train LightGCN/BPR on permitted user-constituent feedback only.

    Explicit side feedback is used when supplied. Otherwise UI is binarized
    training-UB @ catalog-BI. All declared constituent items retain a row.
    No validation/test data is accepted by this function.
    """
    import torch
    from ..bundle_utils.graph import bi_norm, bipartite
    from ..bundle_utils.torch_utils import lightgcn_propagate, sp_to_torch

    if num_epochs < 1 or emb_size < 1 or num_layers < 0 or batch_size < 1:
        raise ValueError("invalid collaborative feature training dimensions/epochs")
    rng = get_rng(seed)
    torch.manual_seed(0 if seed is None else seed)
    ui = binary(user_item_matrix(train_set)).tocsr()
    users, positives = ui.nonzero()
    eligible = np.diff(ui.indptr)[users] < ui.shape[1]
    users, positives = users[eligible], positives[eligible]
    if not len(users):
        raise ValueError("collaborative features need feedback and negative items")
    graph = sp_to_torch(bi_norm(bipartite(ui)), device)
    nu, ni = ui.shape
    embeddings = torch.nn.Embedding(nu + ni, emb_size, device=device)
    torch.nn.init.normal_(embeddings.weight, std=0.1)
    optimizer = torch.optim.Adam(embeddings.parameters(), lr=learning_rate)
    for _ in range(num_epochs):
        order = rng.permutation(len(users))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            u, p = users[indices], positives[indices]
            n = _negative_items(ui, u, rng)
            u = torch.as_tensor(u, dtype=torch.long, device=device)
            p = torch.as_tensor(p + nu, dtype=torch.long, device=device)
            n = torch.as_tensor(n + nu, dtype=torch.long, device=device)
            all_features = lightgcn_propagate(
                graph, embeddings.weight, num_layers,
                layer_weights=[1.0 / (num_layers + 1)] * (num_layers + 1),
            )
            positive = (all_features[u] * all_features[p]).sum(-1)
            negative = (all_features[u] * all_features[n]).sum(-1)
            loss = torch.nn.functional.softplus(negative - positive).mean()
            loss = loss + lambda_reg * sum(
                embeddings(x).square().sum(-1).mean() for x in (u, p, n)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        output = lightgcn_propagate(
            graph, embeddings.weight, num_layers,
            layer_weights=[1.0 / (num_layers + 1)] * (num_layers + 1),
        )[nu:]
    return normalize_rows(output.detach().cpu().numpy())
