# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Numerical checks of DCBR graph helpers and losses."""

import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp

from cornac.models.dcbr.graph import (
    bi_norm,
    bipartite,
    row_norm,
    user_item_matrix,
)
from cornac.models.dcbr.graph import sample_negatives

from ._bundle_checks import toy_evaluation


# ---------------------------------------------------------------- graph helpers
def test_bi_norm_matches_formula():
    a = sp.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    expected = (
        np.diag(1 / np.sqrt([2, 2])) @ a.toarray() @ np.diag(1 / np.sqrt([1, 2, 1]))
    )
    np.testing.assert_allclose(bi_norm(a).toarray(), expected, rtol=1e-6)


def test_row_norm_is_mean_aggregation():
    a = sp.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(
        row_norm(a).toarray(), [[0.5, 0.5, 0], [0, 0, 1], [0, 0, 0]], atol=1e-6
    )


def test_bipartite_blocks():
    a = sp.csr_matrix(np.array([[1.0, 0.0, 2.0]]))
    full = bipartite(a, diag_rows=sp.identity(1)).toarray()
    assert full.shape == (4, 4)
    np.testing.assert_array_equal(full[0, 1:], [1, 0, 2])
    np.testing.assert_array_equal(full[1:, 0], [1, 0, 2])
    assert full[0, 0] == 1


def test_sample_negatives_avoids_positives():
    positives = sp.csr_matrix(np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=float))
    users = np.array([0, 1] * 50)
    negatives = sample_negatives(positives, users, np.random.RandomState(0))
    assert set(negatives[users == 0]) == {3}
    assert set(negatives[users == 1]) == {0}


def test_user_item_fallback_uses_training_bundles():
    with_ui = toy_evaluation(with_user_items=True).train_set
    np.testing.assert_array_equal(
        user_item_matrix(with_ui).toarray(), (with_ui.user_item_matrix.toarray() != 0)
    )
    no_ui = toy_evaluation(with_user_items=False).train_set
    derived = (no_ui.csr_matrix @ no_ui.bundle_item_matrix).toarray() != 0
    np.testing.assert_array_equal(user_item_matrix(no_ui).toarray(), derived)


# ---------------------------------------------------------------- torch helpers
def test_torch_graph_helpers():
    torch = pytest.importorskip("torch")
    from cornac.models.dcbr.torch_utils import (
        bpr_loss,
        info_nce,
        lightgcn_propagate,
        sp_to_torch,
    )

    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    b = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    an = a.numpy() / np.linalg.norm(a.numpy(), axis=1, keepdims=True)
    bn = b.numpy() / np.linalg.norm(b.numpy(), axis=1, keepdims=True)
    logits = an @ bn.T / 0.5
    expected = np.mean(np.log(np.exp(logits).sum(1)) - np.diag(logits))
    assert abs(info_nce(a, b, 0.5).item() - expected) < 1e-5

    pos, neg = torch.tensor([2.0]), torch.tensor([0.0])
    assert abs(bpr_loss(pos, neg).item() - np.log1p(np.exp(-2.0))) < 1e-6

    graph = sp_to_torch(sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]])))
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = lightgcn_propagate(graph, x, 1, layer_weights=[0.5, 0.5])
    np.testing.assert_allclose(out.numpy(), [[0.5, 0.5], [0.5, 0.5]])


def test_dcbr_import_does_not_load_torch():
    script = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] == 'torch':
            raise ImportError('blocked')
sys.meta_path.insert(0, Block())
import cornac.models.dcbr
from cornac.models import BundleRecommender
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


def test_negative_sampling_rejects_full_catalog_user():
    with pytest.raises(ValueError, match="unobserved bundle"):
        sample_negatives(sp.csr_matrix([[1, 1]]), [0], np.random.RandomState(0))


def test_negative_sampling_completes_after_rejection_budget():
    matrix = sp.csr_matrix([[1, 1, 1, 0]])
    negatives = sample_negatives(
        matrix, np.zeros(20, dtype=int), np.random.RandomState(1), max_tries=0
    )
    np.testing.assert_array_equal(negatives, np.full(20, 3))
