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

"""Opt-in Semantic ID evaluation for Cornac Experiment."""

import time
from collections import OrderedDict

import numpy as np

from ..experiment.result import Result
from ..metrics import Recall, NDCG
from ..models.recommender import SemanticIDRecommender
from ..models.semantic_id import (
    SemanticIDCatalog,
    SemanticIDPrediction,
    SemanticIDTrace,
    SemanticIDTraceRecord,
)
from .next_item_evaluation import NextItemEvaluation
from .sid_evaluation import evaluate_semantic_id_trace, _cutoffs
from .sid_resolvers import HistoryCosine, TrainingPopularity


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
    catalog = model.semantic_id_catalog() if catalog is None else catalog
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
            # Candidate filtering changes admission, never rank positions.
            validity = np.asarray([tuple(row) in catalog.address_to_items
                                   for row in prediction.address_sids])
            if not np.array_equal(validity, prediction.catalog_validity):
                # Verify original model validity before applying an explicit
                # restricted universe; otherwise a malformed adapter is hidden.
                from .sid_resolvers import validate_prediction
                validate_prediction(model.semantic_id_catalog(), prediction)
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
    """Evaluate raw Semantic IDs through SID, CCE and explicit item views.

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
                    native.address_codebook_cardinalities,
                    metadata={"source_catalog_hash": native.stable_hash,
                              "admission": "training_items"})
        cutoffs = tuple(dict.fromkeys(int(metric.k) for metric in metrics))
        trace = collect_semantic_id_trace(model, dataset, mode=self.mode,
            n_candidates=max(cutoffs), catalog=catalog,
            exclude_unknowns=self.exclude_unknowns,
            num_train_items=self.train_set.num_items)
        replay = evaluate_semantic_id_trace(catalog, trace, cutoffs, resolvers)
        averages, per_user = OrderedDict(), OrderedDict()
        users = np.asarray([r.user_index for r in trace.records])
        for view, rows in replay["per_query"].items():
            for metric in metrics:
                field = "recall" if isinstance(metric, Recall) else "ndcg"
                values = rows[field][:, cutoffs.index(metric.k)]
                name = f"{view}/{metric.name}"
                grouped = {int(user): values[users == user].tolist() for user in np.unique(users)}
                per_user[name] = grouped
                averages[name] = (float(np.mean([np.mean(v) for v in grouped.values()]))
                                  if user_based and grouped else
                                  float(values.mean()) if len(values) else 0.)
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
            model.transform(self.val_set)
            validation = self._evaluate_split(model, self.val_set, metrics, resolvers, user_based)
        return result, validation
