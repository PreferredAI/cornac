# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Toy bundle data and the shared checks every PyTorch bundle model must pass."""

import tempfile

import numpy as np

from cornac.eval_methods import BundleEvaluation
from cornac.exception import ScoreException


def toy_bundle_data(seed=0, users=12, bundles=10, items=15):
    """Small random catalog and user-bundle splits."""
    rng = np.random.RandomState(seed)
    catalog = []
    for b in range(bundles):
        for i in rng.choice(items, size=3, replace=False):
            catalog.append(("b%d" % b, "i%d" % i))
    pairs = sorted(
        {
            (u, b)
            for u in range(users)
            for b in rng.choice(bundles, size=4, replace=False)
        }
    )
    rows = [("u%d" % u, "b%d" % b, 1.0) for u, b in pairs]
    rng.shuffle(rows)
    n = len(rows)
    train = rows[: n * 6 // 10]
    val = rows[n * 6 // 10 : n * 8 // 10]
    test = rows[n * 8 // 10 :]
    user_items = [
        ("u%d" % u, "i%d" % rng.randint(items), 1.0)
        for u in range(users)
        for _ in range(3)
    ]
    return train, val, test, catalog, user_items


def toy_evaluation(seed=0, with_user_items=True):
    train, val, test, catalog, user_items = toy_bundle_data(seed)
    return BundleEvaluation.from_splits(
        train_data=train,
        val_data=val,
        test_data=test,
        bundle_item_data=catalog,
        user_item_data=user_items if with_user_items else None,
        test_mask="train",
        exclude_unknowns=True,
        seed=seed,
    )


def check_bundle_model(make, evaluation):
    """Run the shared contract checks on ``make()``, a factory of tiny models.

    Covers: fit returns self, NumPy scores of the right shape, validation-only
    checkpoint selection, learning signal on training positives, seed
    determinism, save/load, clone and unknown users.
    """
    train_set, val_set = evaluation.train_set, evaluation.val_set
    model = make()
    assert model.fit(train_set, val_set) is model
    scores = model.score(0)
    assert scores.shape == (train_set.num_items,)
    assert np.isfinite(scores).all()
    assert abs(model.score(0, 1) - scores[1]) < 1e-4
    assert model.U.shape[0] == train_set.num_users
    assert model.V.shape[0] == train_set.num_items
    assert len(model.val_history) >= 1 and model.best_epoch is not None

    # Training positives should outrank the average bundle after training.
    rows, cols = train_set.csr_matrix.nonzero()
    all_scores = model.U @ model.V.T
    assert all_scores[rows, cols].mean() > all_scores.mean()

    again = make().fit(train_set, val_set)
    np.testing.assert_allclose(again.U, model.U, rtol=1e-4, atol=1e-5)

    with tempfile.TemporaryDirectory() as path:
        loaded = type(model).load(model.save(path))
        np.testing.assert_allclose(loaded.score(0), model.score(0), rtol=1e-5)
    clone = model.clone({"emb_size": 4})
    assert clone.emb_size == 4 and clone.seed == model.seed

    try:
        model.score(10**6)
    except ScoreException:
        pass
    else:
        raise AssertionError("unknown user must raise ScoreException")
    try:
        model.score(0, 10**6)
    except ScoreException:
        pass
    else:
        raise AssertionError("unknown bundle must raise ScoreException")

    without_ui = toy_evaluation(with_user_items=False)
    assert make().fit(without_ui.train_set).U.shape[0] == without_ui.train_set.num_users
    return model
