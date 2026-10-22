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

"""Paired SID, expanded CCE and exact-item evaluation from raw traces."""

from collections.abc import Mapping

import numpy as np

from ..models.semantic_id import SemanticIDCatalog, SemanticIDTrace
from .sid_resolvers import resolve_semantic_ids, validate_prediction


def _cutoffs(values):
    values = tuple(values)
    if (not values or any(isinstance(k, (bool, np.bool_)) or
            not isinstance(k, (int, np.integer)) or k <= 0 for k in values) or
            len(set(values)) != len(values)):
        raise ValueError("cutoffs must be distinct positive finite integers")
    return tuple(int(k) for k in values)


def _rank_rows(rank, cutoffs):
    hit = np.asarray([float(rank is not None and rank < k) for k in cutoffs])
    ndcg = hit / np.log2(rank + 2) if rank is not None else hit.copy()
    return hit, ndcg


def evaluate_semantic_id_trace(catalog, trace, cutoffs=(5, 10, 20), resolvers=None):
    """Evaluate every view on the same ordered raw predictions.

    CCE is the analytical expectation under uniform permutations *within full
    expanded buckets*. Earlier buckets consume their full widths. Invalid and
    duplicate addresses consume one empty rank slot in every view. A member
    resolver chooses one item per bucket without changing address order.

    ``resolvers`` maps display names to SemanticIDResolver instances. Callbacks
    receive query/user identity and bucket members only. Results contain scalar
    aggregates, per-query Recall/NDCG arrays and protocol/diagnostic sidecars.
    This function averages over queries; SemanticIDEvaluation also supports the
    framework's per-user aggregation.
    """
    cutoffs = _cutoffs(cutoffs)
    if not isinstance(catalog, SemanticIDCatalog) or not isinstance(trace, SemanticIDTrace):
        raise ValueError("expected SemanticIDCatalog and SemanticIDTrace")
    if trace.catalog_hash != catalog.stable_hash:
        raise ValueError("trace catalog_hash differs from catalog")
    if resolvers is None:
        resolvers = {}
    if not isinstance(resolvers, Mapping) or any(
            not isinstance(name, str) or not name or any(c in name for c in "[]/")
            for name in resolvers):
        raise ValueError("resolvers must map nonempty plain names to resolvers")
    from .sid_resolvers import SemanticIDResolver
    if any(not isinstance(r, SemanticIDResolver) for r in resolvers.values()):
        raise ValueError("resolvers must contain SemanticIDResolver instances")
    views = ("SID", "CCE") + tuple(f"Item[{name}]" for name in resolvers)
    rows = {view: {metric: np.zeros((len(trace.records), len(cutoffs)))
                   for metric in ("recall", "ndcg")} for view in views}
    invalid = duplicate = total = 0
    sizes = []
    for index, record in enumerate(trace.records):
        try:
            address = catalog.address_sid_for_item(record.target_item_index)
            tokenizer = catalog.tokenizer_sid_for_item(record.target_item_index)
        except KeyError as error:
            raise ValueError("trace target item is outside catalog") from error
        if address != record.target_address_sid or tokenizer != record.target_tokenizer_sid:
            raise ValueError("trace target codes differ from catalog")
        prediction = record.prediction
        validate_prediction(catalog, prediction)
        addresses = [tuple(int(v) for v in row) for row in prediction.address_sids]
        rank = next((i for i, value in enumerate(addresses) if value == address), None)
        rows["SID"]["recall"][index], rows["SID"]["ndcg"][index] = _rank_rows(rank, cutoffs)
        before, seen, reached = 0, set(), False
        for value, valid in zip(addresses, prediction.catalog_validity):
            if value == address:
                reached = True
                break
            if not valid or value in seen:
                before += 1
            else:
                seen.add(value)
                before += len(catalog.address_to_items[value])
        size = len(catalog.address_to_items[address])
        sizes.append(size)
        if reached:
            for col, cutoff in enumerate(cutoffs):
                included = min(size, max(0, cutoff - before))
                rows["CCE"]["recall"][index, col] = included / size
                rows["CCE"]["ndcg"][index, col] = (
                    np.sum(1 / np.log2(before + np.arange(included) + 2)) / size)
        for name, resolver in resolvers.items():
            slots = resolve_semantic_ids(catalog, prediction, resolver,
                                         user_index=record.user_index,
                                         query_index=record.query_index)
            item_rank = next((i for i, item in enumerate(slots)
                              if item == record.target_item_index), None)
            rows[f"Item[{name}]"]["recall"][index], rows[f"Item[{name}]"]["ndcg"][index] = _rank_rows(item_rank, cutoffs)
        total += len(addresses)
        invalid += int(np.count_nonzero(~prediction.catalog_validity))
        duplicate += len(addresses) - len(set(addresses))
    aggregate = {}
    for view, metrics in rows.items():
        aggregate[view] = {}
        for metric, values in metrics.items():
            values.setflags(write=False)
            for col, cutoff in enumerate(cutoffs):
                aggregate[view][f"{metric.upper() if metric == 'ndcg' else 'Recall'}@{cutoff}"] = (
                    float(values[:, col].mean()) if len(values) else 0.)
    return {"aggregate": aggregate, "per_query": rows,
            "diagnostics": {"invalid_rate": invalid / total if total else 0.,
                            "duplicate_rate": duplicate / total if total else 0.,
                            "target_collision_group_size": np.asarray(sizes, dtype="int64")},
            "protocol": {"catalog_hash": catalog.stable_hash, "cutoffs": cutoffs,
                         "invalid_and_duplicate_slots": "preserved",
                         "order": "decoder_order", "CCE": "uniform_expanded_bucket_expectation",
                         "resolvers": {name: resolver.metadata for name, resolver in resolvers.items()}}}
