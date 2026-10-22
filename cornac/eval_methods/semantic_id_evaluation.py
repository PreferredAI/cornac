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

"""Next-item recommendation evaluation from raw Semantic ID predictions."""

import time
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from ..experiment.result import Result
from ..metrics import Recall, NDCG
from ..models.recommender import SemanticIDRecommender
from ..utils.semantic_id import (
    SemanticIDCatalog,
    SemanticIDPrediction,
    SemanticIDTrace,
    SemanticIDTraceRecord,
    SemanticIDResolver,
    HistoryCosine,
    TrainingPopularity,
    resolve_semantic_ids,
    validate_prediction,
)
from .next_item_evaluation import NextItemEvaluation


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
    framework's per-user aggregation. CCE follows Zhang et al.,
    "Faithful Evaluation of Semantic-ID Tokenizers for Generative Recommendation"
    (CIKM 2026): https://arxiv.org/abs/2605.25330v2.
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


def collect_semantic_id_trace(model, dataset, *, mode="last", n_candidates=20,
                              catalog=None, exclude_unknowns=False,
                              num_train_items=None):
    """Collect raw predictions once per query, with no target in model context.

    ``n_candidates`` requests a minimum budget; legitimately shorter decoder
    output stays shorter. ``catalog`` may restrict the model's complete catalog
    to the evaluator's admitted universe. Excluded addresses retain empty slots.
    """
    if mode not in ("last", "next"):
        raise ValueError("mode must be last or next")
    _cutoffs((n_candidates,))
    native_catalog = model.semantic_id_catalog()
    catalog = native_catalog if catalog is None else catalog
    records = []
    for [session], [mapped], [items] in dataset.si_iter(batch_size=1, shuffle=False):
        if len(items) < 2:
            continue
        user = int(dataset.uir_tuple[0][mapped[0]])
        start = 1 if mode == "next" else len(items) - 1
        for position in range(start, len(items)):
            target = int(items[position])
            if exclude_unknowns and target >= num_train_items:
                continue
            query = len(records)
            prediction = model.predict_semantic_ids(
                history_items=list(items[:position]), user_index=user,
                query_index=query, session_index=int(session),
                evaluation_position=position, n_candidates=n_candidates)
            validate_prediction(native_catalog, prediction)
            # Candidate filtering changes admission, never rank positions.
            validity = np.asarray([tuple(row) in catalog.address_to_items
                                   for row in prediction.address_sids])
            if not np.array_equal(validity, prediction.catalog_validity):
                prediction = SemanticIDPrediction(prediction.address_sids,
                    prediction.scores, validity, prediction.decoder_diagnostics)
            try:
                address = catalog.address_sid_for_item(target)
                tokenizer = catalog.tokenizer_sid_for_item(target)
            except KeyError as error:
                raise ValueError("evaluation target is absent from the admitted model catalog") from error
            records.append(SemanticIDTraceRecord(query, user, int(session), position,
                0, position, target, tokenizer, address, prediction))
    return SemanticIDTrace(catalog.stable_hash, tuple(records),
                           {"mode": mode, "requested_candidates": n_candidates})


class SemanticIDEvaluation(NextItemEvaluation):
    """Evaluate next-item recommendation through SID, CCE and item views.

    Use this instead of NextItemEvaluation when raw Semantic ID predictions
    can refer to multiple items. NextItemEvaluation scores item rankings from
    ``model.rank()``; this evaluator preserves raw decoder slots and measures
    recommendation performance under CCE or explicit item resolution. It does
    not evaluate the intrinsic quality of the IDs.

    Models must implement SemanticIDRecommender. Evaluate ordinary next-item
    models separately with NextItemEvaluation on identical data splits and
    cutoffs; a single mixed-model Experiment is not supported.


    Accepts NextItemEvaluation dataset/splitting arguments. ``resolvers`` is an
    optional mapping of names to fitted SemanticIDResolver instances; None uses
    training popularity. ``item_features`` explicitly enables training-history
    cosine and must use global Cornac item rows. ``feature_source`` describes
    their provenance. Only Recall/NDCG with positive finite k are supported.
    Existing model ranking/training and checkpoint selection remain unchanged.
    """

    def __init__(self, *args, resolvers=None, item_features=None,
                 feature_source="provided", **kwargs):
        self.resolvers = resolvers
        self.item_features = item_features
        self.feature_source = feature_source
        super().__init__(*args, **kwargs)

    def _evaluate_split(self, model, dataset, metrics, resolvers, user_based):
        native = model.semantic_id_catalog()
        catalog = native
        if self.exclude_unknowns:
            mask = native.item_indices < self.train_set.num_items
            if not np.all(mask):
                catalog = SemanticIDCatalog(native.item_indices[mask],
                    native.tokenizer_sids[mask], native.address_sids[mask],
                    native.tokenizer_codebook_cardinalities,
                    native.address_codebook_cardinalities)
        cutoffs = tuple(dict.fromkeys(int(metric.k) for metric in metrics))
        trace = collect_semantic_id_trace(model, dataset, mode=self.mode,
            n_candidates=max(cutoffs), catalog=catalog,
            exclude_unknowns=self.exclude_unknowns,
            num_train_items=self.train_set.num_items)
        replay = evaluate_semantic_id_trace(catalog, trace, cutoffs, resolvers)
        averages, per_user = OrderedDict(), OrderedDict()
        users = np.asarray([r.user_index for r in trace.records])
        user_queries = {int(user): np.flatnonzero(users == user)
                        for user in np.unique(users)}
        cutoff_columns = {cutoff: column for column, cutoff in enumerate(cutoffs)}
        for view, rows in replay["per_query"].items():
            for metric in metrics:
                field = "recall" if isinstance(metric, Recall) else "ndcg"
                values = rows[field][:, cutoff_columns[metric.k]]
                name = f"{view}/{metric.name}"
                grouped = {user: values[indices].tolist()
                           for user, indices in user_queries.items()}
                per_user[name] = grouped
                if not len(values):
                    averages[name] = 0.
                elif user_based:
                    averages[name] = float(np.mean([np.mean(v) for v in grouped.values()]))
                else:
                    averages[name] = float(values.mean())
        result = Result(model.name, averages, per_user)
        result.semantic_id_trace = trace
        result.semantic_id_evaluation = replay
        return result

    def evaluate(self, model, metrics, user_based=False, show_validation=True):
        if not isinstance(model, SemanticIDRecommender):
            raise ValueError("model must implement SemanticIDRecommender")
        if not metrics or any(type(metric) not in (Recall, NDCG) for metric in metrics):
            raise ValueError("SemanticIDEvaluation supports Recall and NDCG only")
        _cutoffs(tuple(dict.fromkeys(metric.k for metric in metrics)))
        if self.train_set is None or self.test_set is None:
            raise ValueError("train_set and test_set are required")
        self._reset()
        started = time.time()
        model.fit(self.train_set, self.val_set)
        train_time = time.time() - started
        users, items = self.train_set.uir_tuple[:2]
        resolvers = (dict(self.resolvers) if self.resolvers is not None else
                     {"popularity": TrainingPopularity(items, self.total_items)})
        if self.item_features is not None:
            if "history_cosine" in resolvers:
                raise ValueError("history_cosine resolver name is already configured")
            if len(self.item_features) < self.total_items:
                raise ValueError("item_features must cover all global Cornac item indices")
            resolvers["history_cosine"] = HistoryCosine(self.item_features, users,
                items, self.total_users, feature_source=self.feature_source)
        started = time.time()
        model.transform(self.test_set)
        result = self._evaluate_split(model, self.test_set, metrics, resolvers, user_based)
        result.metric_avg_results["Train (s)"] = train_time
        result.metric_avg_results["Test (s)"] = time.time() - started
        validation = None
        if show_validation and self.val_set is not None:
            started = time.time()
            model.transform(self.val_set)
            validation = self._evaluate_split(model, self.val_set, metrics, resolvers, user_based)
            validation.metric_avg_results["Time (s)"] = time.time() - started
        return result, validation
