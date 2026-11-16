# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Bundle/RPG ID boundaries, leakage, selection, and real backbone checks."""

import tempfile

import numpy as np
import pytest

from cornac.eval_methods import BundleEvaluation
from cornac.exception import ScoreException
from cornac.models import BundleRPG
from cornac.models.bundle_sid.adapter import bundle_to_sequential
from cornac.models.bundle_sid.features import (
    _negative_items, collaborative_item_features, mean_bundle_features,
)


@pytest.fixture
def evaluation():
    # 300 train-observed bundles satisfy the fixed 256-code OPQ contract.
    # b300 is catalog-only; u6 has no training history. IDs deliberately differ
    # between bundle and constituent namespaces, and catalogs have fixed order.
    memberships = [("b%d" % b, "i%d" % ((b + j) % 13))
                   for b in range(301) for j in range(3)]
    train = [("u%d" % (b % 6), "b%d" % b, 1.0) for b in range(300)]
    val = [("u%d" % u, "b%d" % ((u + 1) % 6), 1.0) for u in range(6)]
    test = [("u%d" % u, "b300", 1.0) for u in range(7)]
    return BundleEvaluation.from_splits(
        train_data=train, val_data=val, test_data=test,
        bundle_item_data=memberships,
        user_item_data=[("u%d" % u, "i%d" % ((u + j) % 13), 1.0)
                        for u in range(7) for j in range(3)],
        user_ids=["u%d" % u for u in range(7)],
        bundle_ids=["b%d" % b for b in range(301)],
        constituent_item_ids=["i%d" % i for i in range(13)],
        test_mask="train", seed=3,
    )


def test_adapter_preserves_maps_and_training_only_histories(evaluation):
    train = evaluation.train_set
    features = np.random.RandomState(0).normal(size=(301, 8))
    sequential, histories = bundle_to_sequential(train, features, n_perm=3, seed=9)
    assert sequential.iid_map == train.iid_map
    assert sequential.uid_map == train.uid_map
    assert sequential.num_items == 301
    assert 300 not in sequential.uir_tuple[1]
    assert histories[6] == []
    for user, history in enumerate(histories):
        assert set(history) == set(train.csr_matrix.getrow(user).indices)
    again, first = bundle_to_sequential(train, features, n_perm=1, seed=9)
    assert first == histories  # inference order independent of n_perm
    assert len(sequential.uir_tuple[0]) == 3 * len(again.uir_tuple[0])
    np.testing.assert_array_equal(sequential.item_feature.features, features.astype("float32"))
    _, other = bundle_to_sequential(train, features, seed=10)
    assert other != histories
    with pytest.raises(ValueError, match="features"):
        bundle_to_sequential(train, features[:-1])


def test_mean_pooling_and_negative_postcondition(evaluation):
    import scipy.sparse as sp

    features = np.eye(13, dtype=np.float32)
    result = mean_bundle_features(evaluation.train_set, features)
    expected = np.zeros(13)
    expected[[0, 1, 2]] = 1 / np.sqrt(3)
    np.testing.assert_allclose(result[0], expected, atol=1e-6)
    dense = sp.csr_matrix([[1] * 99 + [0]])
    negatives = _negative_items(dense, np.zeros(100, dtype=int), np.random.RandomState(2))
    assert set(negatives) == {99}
    with pytest.raises(ValueError, match="saturated"):
        _negative_items(sp.csr_matrix([[1, 1]]), np.array([0]), np.random.RandomState(2))


def test_cf_features_are_aligned_and_reproducible(evaluation):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    params = dict(emb_size=8, num_epochs=2, batch_size=32, seed=3)
    a = collaborative_item_features(evaluation.train_set, **params)
    b = collaborative_item_features(evaluation.train_set, **params)
    assert a.shape == (13, 8)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b)


def test_clhe_training_and_member_order_invariance(evaluation):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from cornac.models.bundle_sid.clhe_encoder import CLHEBundleEncoder, encode_bundles

    features = np.random.RandomState(3).normal(size=(13, 8)).astype("float32")
    values, diagnostics = encode_bundles(
        evaluation.train_set, features, num_epochs=2, batch_size=64,
        heads=2, max_bundle_items=2, seed=4,
    )
    assert values.shape == (301, 8) and np.isfinite(values).all()
    assert diagnostics["truncated_bundles"] == 301
    assert len(diagnostics["losses"]) == 2
    assert all(np.isfinite(diagnostics["losses"]))
    encoder = CLHEBundleEncoder(8, heads=2).eval()
    x = torch.as_tensor(features[:3]).unsqueeze(0)
    mask = torch.ones((1, 3), dtype=torch.bool)
    with torch.no_grad():
        np.testing.assert_allclose(encoder(x, mask), encoder(x[:, [2, 0, 1]], mask), atol=1e-6)


def test_wrapper_real_backbone_selection_and_serialization(evaluation, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    torch.set_num_threads(1)
    from cornac.models.rpg import RPG

    # The fixture tests the real unchanged neural training/decoding and outer
    # selection. Replace only expensive FAISS fitting with known catalog codes;
    # a separate real-FAISS pilot is required for tokenizer performance claims.
    def tokens(self, feats, train_mask=None):
        assert train_mask.sum() == 300 and not train_mask[300]
        rows = np.arange(len(feats))
        return np.stack((rows % 256, rows // 256), axis=1).astype("int64")
    monkeypatch.setattr(RPG, "_opq_tokenize", tokens)
    features = np.random.RandomState(1).normal(size=(13, 8)).astype("float32")
    model = BundleRPG(
        encoder="mean", emb_size=8, item_features=features,
        epoch_budgets=(1, 2), eval_k=5, seed=3,
        rpg_params=dict(n_codebook=2, d_model=16, n_layer=1, n_head=2,
                        n_inner=32, max_len=4, batch_size=128, graph_edges=2,
                        embd_dropout=0, attn_dropout=0),
    )
    assert model.fit(evaluation.train_set, evaluation.val_set) is model
    assert len(model.val_history) == 2
    assert model.best_value == max(value for _, value in model.val_history)
    assert model.best_epoch in (1, 2)
    scores = model.score(0)
    assert scores.shape == (301,) and np.isfinite(scores).all()
    assert model.score(0, 300) == pytest.approx(scores[300])
    np.testing.assert_array_equal(model.score(6), model.popularity_)
    assert model.rpg_.scoring == "exact" and model.rpg_.model_selection == "last"
    assert model.training_diagnostics_["empty_history_users"] == 1
    with tempfile.TemporaryDirectory() as path:
        loaded = BundleRPG.load(model.save(path))
        np.testing.assert_allclose(loaded.score(0), scores, atol=1e-6)
    clone = model.clone({"encoder": "clhe"})
    assert clone.encoder == "clhe" and clone.seed == 3
    for u, b in [(1000, None), (0, 1000), (-1, None)]:
        with pytest.raises(ScoreException):
            model.score(u, b)


def test_parameter_contract():
    with pytest.raises(ValueError, match="cannot override"):
        BundleRPG(rpg_params={"scoring": "graph"})
    with pytest.raises(ValueError, match="epoch_budgets"):
        BundleRPG(epoch_budgets=(0,))
    with pytest.raises(ValueError, match="encoder"):
        BundleRPG(encoder="cogen")


def test_small_catalog_fails_before_feature_training():
    from ._bundle_checks import toy_evaluation

    with pytest.raises(ValueError, match="256 training-observed"):
        BundleRPG().fit(toy_evaluation().train_set)
