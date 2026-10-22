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

"""Target-free selection of one item within each generated complete SID."""

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from ..models.semantic_id import SemanticIDCatalog, SemanticIDPrediction


def _indices(values, name, upper=None):
    values = np.asarray(values)
    if (values.ndim != 1 or (values.size and
            not np.issubdtype(values.dtype, np.integer)) or
            np.any(values < 0) or (upper is not None and np.any(values >= upper))):
        raise ValueError(f"{name} must contain covered nonnegative integer indices")
    return values.astype("int64", copy=False)


def _unit_rows(features):
    features = np.array(features, dtype="float64", copy=True)
    if (features.ndim != 2 or not all(features.shape) or
            not np.all(np.isfinite(features))):
        raise ValueError("features must be a nonempty finite item matrix")
    scales = np.max(np.abs(features), axis=1, keepdims=True)
    np.divide(features, scales, out=features, where=scales != 0)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    np.divide(features, norms, out=features, where=norms != 0)
    return np.frombuffer(features.tobytes(), dtype=features.dtype).reshape(features.shape)


class SemanticIDResolver(ABC):
    """Score candidate members without targets; highest score wins.

    Implement ``score(query_index, user_index, candidate_items)``. The shared
    operation checks scores and breaks ties by smallest global item index.
    Scores never reorder generated addresses or admit outside candidates.
    """

    @abstractmethod
    def score(self, query_index, user_index, candidate_items):
        """Return one finite score per candidate, in candidate order."""
        raise NotImplementedError

    @property
    def metadata(self):
        return {"resolver": type(self).__name__}


class TrainingPopularity(SemanticIDResolver):
    """Count explicitly supplied training events, retaining repeated events."""

    def __init__(self, training_items, num_items):
        if not isinstance(num_items, (int, np.integer)) or num_items <= 0:
            raise ValueError("num_items must be a positive integer")
        items = _indices(training_items, "training_items", num_items)
        counts = np.bincount(items, minlength=num_items)
        self.counts = np.frombuffer(counts.tobytes(), dtype=counts.dtype)

    def score(self, query_index, user_index, candidate_items):
        return self.counts[list(candidate_items)]

    @property
    def metadata(self):
        return {"resolver": "training_popularity", "source": "training_events",
                "counts_sha256": hashlib.sha256(self.counts.tobytes()).hexdigest()}


class HistoryCosine(SemanticIDResolver):
    """Cosine to normalized sums of normalized training-event item features.

    Rows are global Cornac item indices. Supply training events only; repeats
    contribute repeatedly. A zero user profile ties all members. ``feature_source``
    labels the explicit scoring feature space, which need not be the model's
    tokenizer feature space. No evaluation history or target enters the profile.
    """

    def __init__(self, features, training_users, training_items, num_users,
                 feature_source="provided"):
        self.features = _unit_rows(features)
        if not isinstance(num_users, (int, np.integer)) or num_users <= 0:
            raise ValueError("num_users must be a positive integer")
        users = _indices(training_users, "training_users", num_users)
        items = _indices(training_items, "training_items", len(self.features))
        if users.shape != items.shape:
            raise ValueError("training user/item events must align")
        if not isinstance(feature_source, str) or not feature_source:
            raise ValueError("feature_source must be a nonempty string")
        profiles = np.zeros((num_users, self.features.shape[1]), dtype="float64")
        np.add.at(profiles, users, self.features[items])
        self.profiles = _unit_rows(profiles)
        self.feature_source = feature_source

    def score(self, query_index, user_index, candidate_items):
        if user_index is None or not 0 <= user_index < len(self.profiles):
            raise ValueError("user_index is outside the training profile space")
        return self.features[list(candidate_items)] @ self.profiles[user_index]

    @property
    def metadata(self):
        return {"resolver": "training_history_cosine", "source": "training_events",
                "feature_source": self.feature_source,
                "feature_join": "global_item_index", "zero_profile": "item_index_tie",
                "normalized_features_sha256": hashlib.sha256(self.features.tobytes()).hexdigest()}


class CustomScorer(SemanticIDResolver):
    """Adapt ``callback(query_index, user_index, candidate_items)`` to a resolver."""

    def __init__(self, callback):
        if not callable(callback):
            raise ValueError("callback must be callable")
        self.callback = callback

    def score(self, query_index, user_index, candidate_items):
        return self.callback(query_index, user_index, candidate_items)


def validate_prediction(catalog, prediction):
    """Check raw prediction validity against the admitted frozen catalog."""
    if not isinstance(catalog, SemanticIDCatalog) or not isinstance(prediction, SemanticIDPrediction):
        raise ValueError("expected SemanticIDCatalog and SemanticIDPrediction")
    if prediction.address_sids.shape[1] != catalog.address_sids.shape[1]:
        raise ValueError("prediction address width differs from catalog")
    expected = [tuple(row) in catalog.address_to_items for row in prediction.address_sids]
    if not np.array_equal(prediction.catalog_validity, expected):
        raise ValueError("prediction catalog_validity differs from catalog")


def resolve_semantic_ids(catalog, prediction, resolver, *, user_index=None,
                         query_index=None):
    """Return item/None rank slots from raw predictions for inference or replay.

    Invalid and repeated addresses consume empty slots. Singleton buckets bypass
    scoring. This operation returns no fillers, changes no input, and exposes no
    target to the resolver. Choose a complete-address catalog matching the model.
    """
    validate_prediction(catalog, prediction)
    if not isinstance(resolver, SemanticIDResolver):
        raise ValueError("resolver must be a SemanticIDResolver")
    slots, seen = [], set()
    for row, valid in zip(prediction.address_sids, prediction.catalog_validity):
        address = tuple(int(value) for value in row)
        if not valid or address in seen:
            slots.append(None)
            continue
        seen.add(address)
        items = catalog.address_to_items[address]
        if len(items) == 1:
            slots.append(items[0])
            continue
        scores = np.asarray(resolver.score(query_index, user_index, items), dtype="float64")
        if scores.shape != (len(items),) or not np.all(np.isfinite(scores)):
            raise ValueError("resolver must return one finite score per candidate")
        slots.append(min(zip(items, scores), key=lambda pair: (-pair[1], pair[0]))[0])
    return tuple(slots)
