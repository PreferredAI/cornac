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

import itertools
import pickle
import subprocess
import sys

import numpy as np
import pytest

from cornac import Experiment
from cornac.eval_methods import (SemanticIDEvaluation, evaluate_semantic_id_trace,
    TrainingPopularity, HistoryCosine, CustomScorer, resolve_semantic_ids)
from cornac.metrics import Recall, NDCG, HitRatio
from cornac.models import (SemanticIDCatalog, SemanticIDPrediction,
    SemanticIDRecommender, SemanticIDTrace, SemanticIDTraceRecord)
from cornac.models.semantic_id import prediction_from_catalog_scores


def catalog(codes=((0,), (0,), (1,), (1,), (2,))):
    codes = np.asarray(codes)
    cardinalities = tuple(int(codes[:, i].max()) + 1 for i in range(codes.shape[1]))
    return SemanticIDCatalog(np.arange(len(codes)), codes, codes, cardinalities, cardinalities)


def prediction(cat, addresses):
    values = np.asarray(addresses, dtype="int64").reshape(-1, cat.address_sids.shape[1])
    return SemanticIDPrediction(values, np.arange(len(values), dtype=float),
        [tuple(row) in cat.address_to_items for row in values])


def trace(cat, target, addresses, user=0, query=0):
    return SemanticIDTrace(cat.stable_hash, (SemanticIDTraceRecord(query, user,
        0, 1, 0, 1, target, cat.tokenizer_sid_for_item(target),
        cat.address_sid_for_item(target), prediction(cat, addresses)),))


def test_catalog_freezes_copies_hash_and_pickle():
    codes = np.array([[0], [0], [1]])
    cat = catalog(codes)
    codes[0, 0] = 9
    assert cat.address_to_items[(0,)] == (0, 1)
    with pytest.raises(ValueError):
        cat.address_sids.setflags(write=True)
    loaded = pickle.loads(pickle.dumps(cat))
    assert loaded.stable_hash == cat.stable_hash
    assert loaded.address_to_items == cat.address_to_items
    with pytest.raises(TypeError):
        cat.address_to_items[(0,)] = (1,)
    with pytest.raises(ValueError, match="integers"):
        catalog(((0.5,), (1.,)))


def test_resolution_preserves_slots_members_ties_and_target_free_context():
    cat = catalog()
    pred = prediction(cat, ((8,), (1,), (1,), (0,), (2,)))
    calls = []
    def score(query, user, candidates):
        calls.append((query, user, candidates))
        return np.zeros(len(candidates))
    slots = resolve_semantic_ids(cat, pred, CustomScorer(score), user_index=2, query_index=7)
    assert slots == (None, 2, None, 0, 4)
    assert calls == [(7, 2, (2, 3)), (7, 2, (0, 1))]
    assert pred.address_sids.tolist() == [[8], [1], [1], [0], [2]]
    for bad in (lambda *args: [1], lambda *args: [np.nan, 0]):
        with pytest.raises(ValueError, match="finite score"):
            resolve_semantic_ids(cat, pred, CustomScorer(bad))


def test_training_popularity_and_cosine_multiplicity_and_zero_profile():
    cat = catalog()
    pop = TrainingPopularity([1, 1, 0, 3], 5)
    assert resolve_semantic_ids(cat, prediction(cat, [[0], [1]]), pop) == (1, 3)
    features = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 0]], dtype=float)
    scorer = HistoryCosine(features, [0, 0, 0], [0, 1, 1], 2, "fixed-test-features")
    assert resolve_semantic_ids(cat, prediction(cat, [[0], [1]]), scorer, user_index=0) == (1, 3)
    assert resolve_semantic_ids(cat, prediction(cat, [[0], [1]]), scorer, user_index=1) == (0, 2)
    features[:] = 0
    assert scorer.features[0, 0] == 1
    assert scorer.metadata["feature_source"] == "fixed-test-features"
    with pytest.raises(ValueError, match="align"):
        HistoryCosine(np.ones((5, 2)), [0], [0, 1], 2)
    with pytest.raises(ValueError):
        TrainingPopularity([5], 5)
    with pytest.raises(ValueError):
        HistoryCosine([[np.nan]], [0], [0], 1)


@pytest.mark.parametrize("addresses", [((0,), (1,), (2,)),
    ((9,), (0,), (0,), (1,), (2,)), ((1,), (0,)), (), ((2,),)])
def test_cce_matches_exhaustive_expansion_oracle(addresses):
    cat = catalog()
    buckets, seen = [], set()
    for address in addresses:
        if address not in cat.address_to_items or address in seen:
            buckets.append(((None,),))
        else:
            buckets.append(tuple(itertools.permutations(cat.address_to_items[address])))
            seen.add(address)
    lists = [sum(parts, ()) for parts in itertools.product(*buckets)]
    cutoffs = tuple(range(1, 9))
    for target in cat.item_indices:
        result = evaluate_semantic_id_trace(cat, trace(cat, int(target), addresses), cutoffs)
        for column, cutoff in enumerate(cutoffs):
            hits, ndcgs = [], []
            for items in lists:
                rank = items.index(target) if target in items else None
                hit = float(rank is not None and rank < cutoff)
                hits.append(hit)
                ndcgs.append(hit / np.log2(rank + 2) if rank is not None else 0.)
            assert result["per_query"]["CCE"]["recall"][0, column] == pytest.approx(np.mean(hits))
            assert result["per_query"]["CCE"]["ndcg"][0, column] == pytest.approx(np.mean(ndcgs))


def test_injective_complete_ids_despite_prefix_collisions():
    cat = SemanticIDCatalog([0, 1, 2], [[0], [0], [1]], [[0, 0], [0, 1], [1, 0]], (2,), (2, 2))
    result = evaluate_semantic_id_trace(cat, trace(cat, 1, ((9, 9), (0, 0), (0, 0), (0, 1))),
        (1, 3, 4), {"popularity": TrainingPopularity([0, 0], 3)})
    for metric in ("recall", "ndcg"):
        np.testing.assert_array_equal(result["per_query"]["SID"][metric], result["per_query"]["CCE"][metric])
        np.testing.assert_array_equal(result["per_query"]["SID"][metric], result["per_query"]["Item[popularity]"][metric])


def test_replay_rejects_mismatches_and_bad_cutoffs():
    cat = catalog()
    saved = trace(cat, 1, ((0,),))
    for cutoffs in ((0,), (1.2,), (True,), (), (1, 1), (np.inf,)):
        with pytest.raises(ValueError):
            evaluate_semantic_id_trace(cat, saved, cutoffs)
    other = catalog(((0,), (1,), (1,), (1,), (2,)))
    with pytest.raises(ValueError, match="catalog_hash"):
        evaluate_semantic_id_trace(other, saved)
    pred = SemanticIDPrediction([[0]], [1.], [False])
    with pytest.raises(ValueError, match="validity"):
        resolve_semantic_ids(cat, pred, TrainingPopularity([], 5))


def test_catalog_score_conversion_declares_max_member_aggregation():
    cat = catalog()
    pred = prediction_from_catalog_scores(cat, [1, 3, 2, 0, 2])
    assert pred.address_sids.tolist() == [[0], [1], [2]]
    assert pred.scores.tolist() == [3, 2, 2]


class FakeSID(SemanticIDRecommender):
    def __init__(self):
        super().__init__("FakeSID")
        self.calls = []

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        self.catalog_ = catalog(tuple((i // 2,) for i in range(self.total_items)))
        return self

    def semantic_id_catalog(self):
        return self.catalog_

    def predict_semantic_ids(self, history_items, **context):
        assert set(context) == {"user_index", "query_index", "session_index",
                                "evaluation_position", "n_candidates"}
        self.calls.append((tuple(history_items), context))
        return prediction(self.catalog_, ((9,), (0,), (0,), (1,)))

    def score(self, *args, **kwargs):
        raise AssertionError("raw evaluation must never call score/rank")


def data():
    train = [("u0", "t0", "a", 0), ("u0", "t0", "b", 1),
             ("u1", "t1", "c", 0), ("u1", "t1", "d", 1)]
    test = [("u0", "s0", "a", 2), ("u0", "s0", "b", 3), ("u0", "s0", "a", 4),
            ("u0", "s1", "b", 2), ("u0", "s1", "c", 3),
            ("u1", "s2", "c", 2), ("u1", "s2", "d", 3)]
    return train, test


@pytest.mark.parametrize("mode, count", [("last", 3), ("next", 4)])
@pytest.mark.parametrize("user_based", [False, True])
def test_experiment_collects_once_per_query_with_modes_and_user_means(mode, count, user_based, tmp_path):
    train, test = data()
    evaluation = SemanticIDEvaluation.from_splits(train, test, fmt="USIT", mode=mode,
        item_features=np.eye(4), feature_source="identity-features")
    model = FakeSID()
    experiment = Experiment(evaluation, [model], [Recall(4), NDCG(4)],
                            user_based=user_based, show_validation=False, verbose=False,
                            save_dir=str(tmp_path))
    experiment.run()
    assert len(model.calls) == count
    assert all(context["n_candidates"] == 4 for _, context in model.calls)
    result = experiment.result[0]
    assert "Item[history_cosine]/Recall@4" in result.metric_avg_results
    replay = result.semantic_id_evaluation
    for view, rows in replay["per_query"].items():
        for field, metric in [("recall", "Recall"), ("ndcg", "NDCG")]:
            values = rows[field][:, 0]
            users = [record.user_index for record in result.semantic_id_trace.records]
            expected = np.mean([np.mean(values[np.array(users) == user]) for user in set(users)]) if user_based else values.mean()
            assert result.metric_avg_results[f"{view}/{metric}@4"] == pytest.approx(expected)
    loaded = FakeSID.load(next(tmp_path.rglob("*.pkl")))
    assert loaded.semantic_id_catalog().stable_hash == model.semantic_id_catalog().stable_hash


def test_evaluation_uses_training_only_and_separates_validation():
    train, test = data()
    val = [("u0", "v0", "c", 5), ("u0", "v0", "b", 6)]
    evaluation = SemanticIDEvaluation.from_splits(train, test, val, fmt="USIT")
    model = FakeSID()
    result, validation = evaluation.evaluate(model, [Recall(4)], show_validation=True)
    expected = TrainingPopularity([0, 1, 2, 3], 4).metadata
    assert result.semantic_id_evaluation["protocol"]["resolvers"]["popularity"] == expected
    assert validation.semantic_id_evaluation["protocol"]["resolvers"]["popularity"] == expected
    assert len(model.calls) == 4
    assert len(validation.semantic_id_trace.records) == 1


@pytest.mark.parametrize("exclude", [False, True])
def test_unknown_item_handling_and_short_predictions(exclude):
    train, test = data()
    test += [("u1", "s3", "a", 2), ("u1", "s3", "new", 3)]
    evaluation = SemanticIDEvaluation.from_splits(train, test, fmt="USIT", exclude_unknowns=exclude)
    model = FakeSID()
    result, _ = evaluation.evaluate(model, [Recall(8)], show_validation=False)
    assert len(result.semantic_id_trace.records) == (3 if exclude else 4)
    assert all(len(record.prediction.address_sids) == 4 for record in result.semantic_id_trace.records)
    assert result.semantic_id_evaluation["diagnostics"]["invalid_rate"] > 0


def test_unsupported_metric_fails_before_fit():
    train, test = data()
    evaluation = SemanticIDEvaluation.from_splits(train, test, fmt="USIT")
    for metrics in ([HitRatio(3)], [Recall(-1)], [Recall(1.5)]):
        model = FakeSID()
        with pytest.raises(ValueError):
            evaluation.evaluate(model, metrics)
        assert not hasattr(model, "catalog_")


def test_lazy_imports():
    subprocess.run([sys.executable, "-c", "import sys; from cornac.eval_methods import SemanticIDEvaluation; from cornac.models import SemanticIDRecommender; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], check=True)


def test_trace_pickle_keeps_identity_and_readonly_predictions():
    cat = catalog()
    saved = trace(cat, 1, ((0,), (9,)))
    restored = pickle.loads(pickle.dumps(saved))
    assert restored.catalog_hash == saved.catalog_hash
    with pytest.raises(ValueError):
        restored.records[0].prediction.address_sids.setflags(write=True)
    result = evaluate_semantic_id_trace(cat, restored, (1, 2))
    assert result["aggregate"]["CCE"]["Recall@1"] == .5


def test_custom_resolver_in_experiment_and_empty_output():
    train, test = data()
    custom = CustomScorer(lambda query, user, items: np.arange(len(items)))
    evaluation = SemanticIDEvaluation.from_splits(train, test, fmt="USIT",
                                                  resolvers={"custom": custom})
    result, _ = evaluation.evaluate(FakeSID(), [Recall(4)], show_validation=False)
    assert "Item[custom]/Recall@4" in result.metric_avg_results
    assert "Item[popularity]/Recall@4" not in result.metric_avg_results
    class EmptySID(FakeSID):
        def predict_semantic_ids(self, history_items, **context):
            return prediction(self.catalog_, ())
    result, _ = evaluation.evaluate(EmptySID(), [Recall(4)], show_validation=False)
    assert all(value == 0 for key, value in result.metric_avg_results.items()
               if key.endswith("@4"))


def test_feature_coverage_rejected_and_trace_query_identity_unique():
    train, test = data()
    evaluation = SemanticIDEvaluation.from_splits(train, test, fmt="USIT",
                                                  item_features=np.eye(2))
    with pytest.raises(ValueError, match="cover"):
        evaluation.evaluate(FakeSID(), [Recall(4)])
    cat = catalog()
    saved = trace(cat, 1, ((0,),))
    with pytest.raises(ValueError, match="unique"):
        SemanticIDTrace(cat.stable_hash, saved.records * 2)


def test_restricted_catalog_keeps_excluded_address_as_empty_slot():
    train, test = data()
    class ExtraCatalogSID(FakeSID):
        def fit(self, train_set, val_set=None):
            super().fit(train_set, val_set)
            self.catalog_ = catalog(((0,), (1,), (2,), (3,), (4,)))
            return self

        def predict_semantic_ids(self, history_items, **context):
            return prediction(self.catalog_, ((4,), (0,)))
    evaluation = SemanticIDEvaluation.from_splits(train, test, fmt="USIT", exclude_unknowns=True)
    result, _ = evaluation.evaluate(ExtraCatalogSID(), [Recall(1), Recall(2)], show_validation=False)
    records = result.semantic_id_trace.records
    assert records[0].prediction.address_sids.tolist() == [[4], [0]]
    assert records[0].prediction.catalog_validity.tolist() == [False, True]
    assert result.semantic_id_evaluation["per_query"]["CCE"]["recall"][0].tolist() == [0., 1.]
    for values in (records[0].prediction.address_sids, records[0].prediction.scores,
                   records[0].prediction.catalog_validity):
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_wrong_target_codes_rejected_and_same_user_queries_are_distinct():
    from dataclasses import replace
    cat = catalog()
    saved = trace(cat, 1, ((0,),))
    record = replace(saved.records[0], target_address_sid=(1,))
    with pytest.raises(ValueError, match="target codes"):
        evaluate_semantic_id_trace(cat, SemanticIDTrace(cat.stable_hash, (record,)))
    seen = []
    resolver = CustomScorer(lambda query, user, candidates:
        seen.append((query, user)) or ([0, 1] if query == 0 else [1, 0]))
    records = saved.records + (replace(saved.records[0], query_index=1),)
    result = evaluate_semantic_id_trace(cat,
        SemanticIDTrace(cat.stable_hash, records), (1,), {"custom": resolver})
    assert seen == [(0, 0), (1, 0)]
    assert result["per_query"]["Item[custom]"]["recall"].tolist() == [[1.], [0.]]
