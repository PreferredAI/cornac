# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Train-only bundle histories adapted to RPG's sequential item interface."""

from collections import OrderedDict

import numpy as np

from ...data import FeatureModality, SequentialDataset
from ...utils import get_rng


def bundle_histories(train_set, seed=None):
    """Return one seeded ordering per user, containing training bundles only.

    The input has no observed chronology. These orders are synthetic and remain
    fixed for inference; validation/test interactions never enter the histories.
    """
    rng = get_rng(seed)
    matrix = train_set.csr_matrix
    histories = []
    for user in range(train_set.num_users):
        ids = matrix.getrow(user).indices.copy()
        rng.shuffle(ids)
        histories.append(ids.tolist())
    return histories


def bundle_to_sequential(train_set, features, n_perm=1, seed=None):
    """Build USIT sessions with the exact outer user and bundle ID mappings.

    Returns the sequential dataset and the fixed first ordering per user.
    Additional permutations augment training only. Full catalog maps retain
    bundles that have features but no training feedback.
    """
    if not isinstance(n_perm, (int, np.integer)) or n_perm < 1:
        raise ValueError("n_perm must be a positive integer")
    features = np.asarray(features, dtype=np.float32)
    if (features.ndim != 2 or features.shape[0] != len(train_set.iid_map)
            or features.shape[1] == 0 or not np.isfinite(features).all()):
        raise ValueError("features must be finite [total_bundles, feature_dim]")
    histories = bundle_histories(train_set, seed)
    rng = get_rng(seed)
    user_ids, bundle_ids = train_set.user_ids, train_set.item_ids
    rows = []
    for user, history in enumerate(histories):
        for permutation in range(n_perm):
            ordered = list(history)
            if permutation:
                rng.shuffle(ordered)
            rows.extend(
                (user_ids[user], (user, permutation), bundle_ids[bundle], step)
                for step, bundle in enumerate(ordered)
            )
    if not any(len(history) >= 2 for history in histories):
        raise ValueError("RPG needs at least one user with two training bundles")
    dataset = SequentialDataset.build(
        rows, fmt="USIT", seed=seed,
        global_uid_map=OrderedDict(train_set.uid_map),
        global_iid_map=OrderedDict(train_set.iid_map),
    )
    dataset.add_modalities(
        item_feature=FeatureModality(features=features.copy(), ids=list(bundle_ids))
    )
    return dataset, histories
