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
"""Graph construction and negative sampling for DCBR (no PyTorch needed)."""

import numpy as np
import scipy.sparse as sp


def binary(matrix):
    """Binary float64 CSR copy of a sparse matrix (None stays None)."""
    if matrix is None:
        return None
    out = (sp.csr_matrix(matrix) != 0).astype(np.float64).tocsr()
    out.eliminate_zeros()
    return out


def user_item_matrix(train_set):
    """Training user x constituent item matrix of a :obj:`BundleDataset`.

    Uses the supplied ``user_item_matrix`` when available, otherwise derives
    it from the user's training bundles (UB @ BI, binarized). Only training
    observations are used, so no validation or test signal can leak.
    """
    if train_set.user_item_matrix is not None:
        return binary(train_set.user_item_matrix)
    return binary(binary(train_set.csr_matrix) @ binary(train_set.bundle_item_matrix))


def bipartite(matrix, diag_rows=None, diag_cols=None):
    """Symmetric block matrix [[R, A], [A^T, C]] of an m x n matrix A.

    ``diag_rows`` (m x m) and ``diag_cols`` (n x n) fill the diagonal blocks,
    zero when None.
    """
    m, n = matrix.shape
    top_left = sp.csr_matrix((m, m)) if diag_rows is None else diag_rows
    bottom_right = sp.csr_matrix((n, n)) if diag_cols is None else diag_cols
    return sp.bmat([[top_left, matrix], [matrix.T, bottom_right]], format="csr")


def bi_norm(matrix, eps=1e-8):
    """Symmetric normalization D_r^-1/2 A D_c^-1/2 (rows/columns by degree)."""
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    rows = np.asarray(matrix.sum(axis=1)).ravel()
    cols = np.asarray(matrix.sum(axis=0)).ravel()
    left = sp.diags(1.0 / (np.sqrt(rows) + eps))
    right = sp.diags(1.0 / (np.sqrt(cols) + eps))
    return (left @ matrix @ right).tocsr()


def row_norm(matrix, eps=1e-8):
    """Row normalization D_r^-1 A, i.e. mean aggregation over each row's neighbors."""
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    rows = np.asarray(matrix.sum(axis=1)).ravel()
    return (sp.diags(1.0 / (rows + eps)) @ matrix).tocsr()


def _observed(keys, query):
    positions = np.searchsorted(keys, query)
    hit = positions < len(keys)
    hit[hit] = keys[positions[hit]] == query[hit]
    return hit


def sample_negatives(positives, users, rng, keys, max_tries=100):
    """Uniform negatives: one column of ``positives`` not observed for each user.

    ``keys`` contains sorted user * num_cols + bundle IDs, built once per fit.
    Vectorized rejection sampling uses binary searches against these keys.
    """
    num_cols = positives.shape[1]
    users = np.asarray(users, dtype=np.int64)
    if np.any(positives.indptr[users + 1] - positives.indptr[users] == num_cols):
        raise ValueError(
            "each training user needs an unobserved bundle for BPR sampling"
        )
    negatives = rng.randint(num_cols, size=len(users))
    for _ in range(max_tries):
        hit = _observed(keys, users * num_cols + negatives)
        if not hit.any():
            break
        negatives[hit] = rng.randint(num_cols, size=int(hit.sum()))
    # Rejection sampling can exhaust its budget for dense users. Finish with an
    # exact uniform draw from their complement rather than returning positives.
    hit = _observed(keys, users * num_cols + negatives)
    for user in np.unique(users[hit]):
        observed = positives.indices[
            positives.indptr[user] : positives.indptr[user + 1]
        ]
        candidates = np.setdiff1d(np.arange(num_cols), observed, assume_unique=True)
        mask = hit & (users == user)
        negatives[mask] = rng.choice(candidates, size=int(mask.sum()))
    return negatives
