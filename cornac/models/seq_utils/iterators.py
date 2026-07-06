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
"""
Mini-batch iterators for session-based training.
"""

from collections import Counter

import numpy as np

from ...utils.common import get_rng


def build_neg_sampler(uir_tuple, sample_alpha):
    """Precompute a popularity-based sampling distribution over items.

    Parameters
    ----------
    uir_tuple : tuple
        ``(user_ids, item_ids, ratings)`` arrays; only ``item_ids`` is used.
    sample_alpha : float
        Popularity smoothing exponent. ``0`` gives a uniform distribution
        (over observed items) and ``1`` gives raw popularity weighting.

    Returns
    -------
    item_indices : numpy.ndarray, shape (num_items,), dtype int
        Item ids ordered from most to least popular.
    item_dist : numpy.ndarray, shape (num_items,), dtype float
        Sampling probabilities aligned with ``item_indices`` (sums to 1).
    """
    item_count = Counter(uir_tuple[1])
    item_indices = np.array([iid for iid, _ in item_count.most_common()], dtype="int")
    item_dist = np.array([cnt for _, cnt in item_count.most_common()], dtype="float") ** sample_alpha
    item_dist = item_dist / item_dist.sum()
    return item_indices, item_dist


def io_iter(s_iter, uir_tuple, n_sample=0, sample_alpha=0, rng=None, batch_size=1, shuffle=False):
    """Session-based per-item iterator (parallel sessions).

    Yields per training step a 4-tuple
    ``(in_iids, out_iids, start_mask, valid_id)`` where:

    - ``in_iids``: current input item id in each slot. Shape ``(B',)``.
    - ``out_iids``: target item ids (followed by ``n_sample`` shared
        negatives). Shape ``(B' + N,)``.
    - ``start_mask``: 1 in slots that just started a new session (used to
        reset RNN hidden state). Shape ``(B',)``.
    - ``valid_id``: indices of the slots that remain valid in the current
        batch (used to trim the hidden state when sessions end at different
        times). Shape ``(B',)``.

    ``B'`` equals ``batch_size`` for the main loop and shrinks during the
    drain phase as sessions are exhausted.
    """
    rng = rng if rng is not None else get_rng(None)
    start_mask = np.zeros(batch_size, dtype="int")
    end_mask = np.ones(batch_size, dtype="int")
    input_iids = None
    output_iids = None
    l_pool = []  # pending sessions (list of mapped-id lists)
    c_pool = [None for _ in range(batch_size)]
    sizes = np.zeros(batch_size, dtype="int")
    if n_sample > 0:
        item_indices, item_dist = build_neg_sampler(uir_tuple, sample_alpha)

    for _, batch_mapped_ids in s_iter(batch_size, shuffle):
        l_pool += batch_mapped_ids
        while len(l_pool) > 0:
            if end_mask.sum() == 0:
                input_iids = uir_tuple[1][[mapped_ids[-sizes[idx]] for idx, mapped_ids in enumerate(c_pool)]]
                output_iids = uir_tuple[1][[mapped_ids[-sizes[idx] + 1] for idx, mapped_ids in enumerate(c_pool)]]
                sizes -= 1
                for idx, size in enumerate(sizes):
                    if size == 1:
                        end_mask[idx] = 1
                if n_sample > 0:
                    negatives = rng.choice(item_indices, size=n_sample, replace=True, p=item_dist)
                    output_iids = np.concatenate([output_iids, negatives])
                yield (
                    input_iids,
                    output_iids,
                    start_mask.copy(),
                    np.arange(batch_size, dtype="int"),
                )
                start_mask.fill(0)
            while end_mask.sum() > 0 and len(l_pool) > 0:
                next_seq = l_pool.pop()
                if len(next_seq) > 1:
                    idx = np.nonzero(end_mask)[0][0]
                    end_mask[idx] = 0
                    start_mask[idx] = 1
                    c_pool[idx] = next_seq
                    sizes[idx] = len(c_pool[idx])

    valid_id = np.ones(batch_size, dtype="int")
    while True:
        for idx, size in enumerate(sizes):
            if size == 1:
                end_mask[idx] = 1
                valid_id[idx] = 0
        keep = [idx for idx in range(len(c_pool)) if sizes[idx] > 1]
        if not keep:
            break
        input_iids = uir_tuple[1][[c_pool[idx][-sizes[idx]] for idx in keep]]
        output_iids = uir_tuple[1][[c_pool[idx][-sizes[idx] + 1] for idx in keep]]
        sizes -= 1
        for idx, size in enumerate(sizes):
            if size == 1:
                end_mask[idx] = 1
        keep_mask = np.nonzero(valid_id)[0]
        start_mask = start_mask[keep_mask]
        end_mask = end_mask[keep_mask]
        sizes = sizes[keep_mask]
        c_pool = [_ for _, valid in zip(c_pool, valid_id) if valid > 0]
        if n_sample > 0:
            negatives = rng.choice(item_indices, size=n_sample, replace=True, p=item_dist)
            output_iids = np.concatenate([output_iids, negatives])
        yield input_iids, output_iids, start_mask.copy(), np.nonzero(valid_id)[0]
        valid_id = np.ones(len(input_iids), dtype="int")
        if end_mask.sum() == len(input_iids):
            break
        start_mask.fill(0)


def session_seq_iter(
    train_set,
    pad_index,
    batch_size=64,
    max_len=20,
    n_sample=2048,
    sample_alpha=0.5,
    rng=None,
    shuffle=True,
):
    """Session-based sequence iterator for transformer/seq models.

    Iterates over sessions (each session = one training sequence). For a
    session ``[i0, i1, ..., iT]`` it yields ``num_sessions * (T)`` training
    triples ``(uid, hist[max_len], target)`` where ``hist`` is the
    left-padded prefix and ``target`` is the next item.
    """
    rng = rng if rng is not None else get_rng(None)
    uir_tuple = train_set.uir_tuple
    sessions = train_set.sessions
    sids = list(sessions.keys())
    if shuffle:
        rng.shuffle(sids)
    if n_sample > 0:
        item_indices, item_dist = build_neg_sampler(uir_tuple, sample_alpha)

    buffer_uids, buffer_hist, buffer_target = [], [], []
    for sid in sids:
        mapped_ids = sessions[sid]
        items = list(uir_tuple[1][mapped_ids])
        if len(items) < 2:
            continue
        uid = int(uir_tuple[0][mapped_ids[0]])
        for t in range(1, len(items)):
            hist = items[:t][-max_len:]
            hist = [pad_index] * (max_len - len(hist)) + list(hist)
            buffer_uids.append(uid)
            buffer_hist.append(hist)
            buffer_target.append(items[t])
            if len(buffer_uids) == batch_size:
                target = np.array(buffer_target, dtype="int")
                if n_sample > 0:
                    negatives = rng.choice(item_indices, size=n_sample, replace=True, p=item_dist)
                    out_iids = np.concatenate([target, negatives])
                else:
                    out_iids = target
                yield (
                    np.array(buffer_uids, dtype="int"),
                    np.array(buffer_hist, dtype="int"),
                    out_iids,
                )
                buffer_uids, buffer_hist, buffer_target = [], [], []
    if len(buffer_uids) > 1:
        target = np.array(buffer_target, dtype="int")
        if n_sample > 0:
            negatives = rng.choice(item_indices, size=n_sample, replace=True, p=item_dist)
            out_iids = np.concatenate([target, negatives])
        else:
            out_iids = target
        yield (
            np.array(buffer_uids, dtype="int"),
            np.array(buffer_hist, dtype="int"),
            out_iids,
        )


def padded_session_iter(
    train_set,
    pad_index,
    batch_size=64,
    max_len=20,
    rng=None,
    shuffle=True,
):
    """Whole-session iterator for transformer next-item models.

    Iterates over sessions, yielding each session as ONE left-padded row
    (no prefix breakdown, no target split, no negative sampling). Training
    objectives (CLM/MLM/PLM/RTD) derive their own targets from the raw
    padded sessions downstream.

    For a session ``[i0, i1, ..., iT]`` the row keeps the last ``max_len``
    items (the head is truncated when longer) and is left-padded with
    ``pad_index`` to exactly ``max_len``. Sessions with fewer than 2 items
    are skipped.

    Parameters
    ----------
    train_set : :class:`~cornac.data.SequentialDataset`
        Must expose ``uir_tuple`` and ``sessions``.
    pad_index : int
        Padding token used to left-pad short sessions.
    batch_size : int, default 64
        Number of sessions per yielded batch.
    max_len : int, default 20
        Fixed sequence length of every row (truncate head / left-pad).
    rng : numpy.random.RandomState, optional
        Random state used to shuffle session order. Defaults to a fresh one.
    shuffle : bool, default True
        Whether to shuffle the session order each epoch.

    Yields
    ------
    uids : numpy.ndarray, shape (B,), dtype int
        User id of each session in the batch.
    padded_seqs : numpy.ndarray, shape (B, max_len), dtype int
        Left-padded, head-truncated item-id sequences. ``B == batch_size``
        for full batches; the final partial batch is yielded whenever it is
        non-empty (``B >= 1``).
    """
    rng = rng if rng is not None else get_rng(None)
    uir_tuple = train_set.uir_tuple
    sessions = train_set.sessions
    sids = list(sessions.keys())
    if shuffle:
        rng.shuffle(sids)

    buffer_uids, buffer_seqs = [], []
    for sid in sids:
        mapped_ids = sessions[sid]
        items = list(uir_tuple[1][mapped_ids])
        if len(items) < 2:
            continue
        uid = int(uir_tuple[0][mapped_ids[0]])
        seq = items[-max_len:]
        seq = [pad_index] * (max_len - len(seq)) + list(seq)
        buffer_uids.append(uid)
        buffer_seqs.append(seq)
        if len(buffer_uids) == batch_size:
            yield (
                np.array(buffer_uids, dtype="int"),
                np.array(buffer_seqs, dtype="int"),
            )
            buffer_uids, buffer_seqs = [], []
    if len(buffer_uids) >= 1:
        yield (
            np.array(buffer_uids, dtype="int"),
            np.array(buffer_seqs, dtype="int"),
        )
