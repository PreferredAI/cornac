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

"""Three-view next-item evaluation with backend-free fixed predictions."""

import itertools
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from cornac.eval_methods import NextItemEvaluation
from cornac.eval_methods.next_item_evaluation import (
    _semantic_metric_scores,
    ranking_eval,
)
from cornac.exception import ScoreException
from cornac.experiment.result import ExperimentResult
from cornac.metrics import Recall, NDCG, AUC, MAP, MRR
from cornac.models import NextItemRecommender, SemanticIDMixin, SPop


class FixedSID(NextItemRecommender, SemanticIDMixin):
    def __init__(self, codes=(0, 0, 1), predicted=(0, 1), scores=(3.0, 2.0, 1.0)):
        super().__init__("FixedSID")
        self.codes = np.asarray(codes, dtype="int64").reshape(-1, 1)
        self.predicted = np.asarray(predicted, dtype="int64").reshape(-1, 1)
        self.scores = np.asarray(scores)
        self.calls = []

    def semantic_id_catalog(self):
        return self.codes

    def _score_with_semantic_ids(self, user_idx, history_items, **kwargs):
        self.calls.append((list(history_items), kwargs))
        return self.scores.copy(), self.predicted.copy()

    def score(self, user_idx, history_items, **kwargs):
        return self._score_with_semantic_ids(user_idx, history_items, **kwargs)[0]


def evaluation(**kwargs):
    train = [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)]
    test = [("u", "te", "a", 3), ("u", "te", "b", 4)]
    kwargs.setdefault("semantic_id_metrics", True)
    return NextItemEvaluation.from_splits(train, test, fmt="USIT", **kwargs)


@pytest.mark.parametrize("entrypoint", ["evaluate", "eval", "ranking_eval"])
@pytest.mark.parametrize("flag", ["omitted", False, True])
@pytest.mark.parametrize("model_class", [FixedSID, SPop])
def test_semantic_views_require_explicit_opt_in(entrypoint, flag, model_class):
    data = evaluation()
    options = {} if flag == "omitted" else {"semantic_id_metrics": flag}
    ev = NextItemEvaluation.from_splits(
        [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)],
        [("u", "te", "a", 3), ("u", "te", "b", 4)],
        fmt="USIT",
        **options,
    )
    model = model_class()
    enabled = flag is True and model_class is FixedSID
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if entrypoint == "evaluate":
            result, _ = ev.evaluate(model, [Recall(1)], False)
            values = list(result.metric_avg_results.values())[:-2]
            names = result.metric_avg_results
        else:
            model.fit(data.train_set)
            if entrypoint == "eval":
                result = NextItemEvaluation.eval(
                    model, data.train_set, data.test_set, True, [Recall(1)], **options
                )
                values = list(result.metric_avg_results.values())
                names = result.metric_avg_results
            else:
                values, _ = ranking_eval(
                    model, [Recall(1)], data.train_set, data.test_set, **options
                )
                names = None
    assert len(values) == (3 if enabled else 1)
    if enabled:
        assert values == [0.0, 1.0, 0.5]
    if names is not None:
        assert ("SID-Recall@1" in names) is enabled
        assert ("CCE-Recall@1" in names) is enabled
    assert len(caught) == int(model_class is FixedSID and not enabled)
    if caught:
        assert "disabled" in str(caught[0].message)


@pytest.mark.parametrize("predicted", [(0, 1, 2), (1, 0), (9, 0, 0, 2, 1), ()])
def test_cce_matches_all_bucket_permutations(predicted):
    groups = {0: (0, 1), 1: (2,), 2: (3, 4, 5)}
    compact = list(dict.fromkeys(sid for sid in predicted if sid in groups))
    rankings = [
        sum((list(bucket) for bucket in permutation), [])
        for permutation in itertools.product(
            *(list(itertools.permutations(groups[sid])) for sid in compact)
        )
    ]
    for target in range(6):
        target_sid = next(sid for sid, members in groups.items() if target in members)
        for cutoff in range(1, 8):
            values, distinct = _semantic_metric_scores(
                np.asarray(predicted).reshape(-1, 1),
                (target_sid,),
                {(sid,): len(members) for sid, members in groups.items()},
                [Recall(cutoff), NDCG(cutoff)],
            )
            oracle_hits, oracle_ndcg = [], []
            for ranking in rankings:
                rank = ranking.index(target) if target in ranking else None
                hit = float(rank is not None and rank < cutoff)
                oracle_hits.append(hit)
                oracle_ndcg.append(hit / np.log2(rank + 2) if rank is not None else 0.0)
            assert values[2:] == pytest.approx(
                [np.mean(oracle_hits), np.mean(oracle_ndcg)]
            )
            assert distinct == len(compact)


def test_three_views_keep_ordinary_resolution_and_share_inference():
    model = FixedSID()
    result, _ = evaluation().evaluate(model, [Recall(1), NDCG(1)], False)
    values = result.metric_avg_results
    assert values["Recall@1"] == 0.0
    assert values["SID-Recall@1"] == 1.0
    assert values["CCE-Recall@1"] == 0.5
    assert values["CCE-NDCG@1"] == 0.5
    assert len(model.calls) == 1
    assert model.calls[0][0] == [0]
    assert {
        "history_mapped_ids",
        "sessions",
        "session_indices",
        "extra_data",
    } <= model.calls[0][1].keys()
    assert not hasattr(result, "semantic_id_trace")


def test_missing_candidates_cannot_get_credit_from_score_fillers():
    model = FixedSID(predicted=(1,), scores=(-1e10, -1e10, 3.0))
    with pytest.warns(UserWarning, match="1 evaluation queries returned fewer than 2"):
        result, _ = evaluation().evaluate(model, [Recall(2)], False)
    assert result.metric_avg_results["SID-Recall@2"] == 0.0
    assert result.metric_avg_results["CCE-Recall@2"] == 0.0


def test_invalid_and_duplicate_predictions_compact_before_cutoff():
    model = FixedSID(predicted=(9, 1, 1, 0, 0))
    result, _ = evaluation().evaluate(model, [Recall(2)], False)
    assert result.metric_avg_results["SID-Recall@2"] == 1.0
    assert result.metric_avg_results["CCE-Recall@2"] == 0.5


@pytest.mark.parametrize("exclude", [False, True])
def test_candidate_filtering_changes_bucket_sizes_and_removes_unknown_only_ids(exclude):
    ev = NextItemEvaluation.from_splits(
        [("u", "tr", "a", 0), ("u", "tr", "b", 1)],
        [("u", "te", "x", 2), ("u", "te", "a", 3), ("u", "te", "b", 4)],
        fmt="USIT",
        exclude_unknowns=False,
        semantic_id_metrics=True,
    )
    # Keep the full catalog available, then restrict the evaluated candidates.
    ev.exclude_unknowns = exclude
    model = FixedSID(codes=(0, 0, 0), predicted=(0,), scores=(3.0, 2.0, 1.0))
    result, _ = ev.evaluate(model, [Recall(1)], False)
    assert result.metric_avg_results["CCE-Recall@1"] == pytest.approx(
        1 / (2 if exclude else 3)
    )
    # An unknown-only bucket consumes no slot when excluded.
    model = FixedSID(codes=(0, 0, 1), predicted=(1, 0))
    result, _ = ev.evaluate(model, [Recall(1)], False)
    assert result.metric_avg_results["SID-Recall@1"] == float(exclude)


@pytest.mark.parametrize("mode,count", [("last", 1), ("next", 2)])
@pytest.mark.parametrize("user_based", [False, True])
def test_modes_user_aggregation_and_validation_share_query_counts(
    mode, count, user_based
):
    train = [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)]
    test = [("u", "te", "a", 3), ("u", "te", "b", 4), ("u", "te", "c", 5)]
    val = [("u", "va", "a", 6), ("u", "va", "b", 7)]
    ev = NextItemEvaluation.from_splits(
        train, test, val, fmt="USIT", mode=mode, semantic_id_metrics=True
    )
    model = FixedSID()
    test_result, val_result = ev.evaluate(model, [Recall(1)], user_based)
    assert len(model.calls) == count + 1
    assert "CCE-Recall@1" in test_result.metric_avg_results
    assert val_result.metric_avg_results["CCE-Recall@1"] == 0.5
    assert model.calls[-1][0] == [0]


def test_user_weighting_matches_ordinary_and_extra_views():
    ev = NextItemEvaluation.from_splits(
        [("u", "tr1", "a", 0), ("u", "tr1", "b", 1), ("v", "tr2", "c", 2)],
        [
            ("u", "te1", "a", 3),
            ("u", "te1", "b", 4),
            ("u", "te2", "a", 5),
            ("u", "te2", "b", 6),
            ("v", "te3", "a", 7),
            ("v", "te3", "c", 8),
        ],
        fmt="USIT",
        semantic_id_metrics=True,
    )
    query, _ = ev.evaluate(FixedSID(), [Recall(1)], False)
    user, _ = ev.evaluate(FixedSID(), [Recall(1)], True)
    assert query.metric_avg_results["SID-Recall@1"] == pytest.approx(2 / 3)
    assert user.metric_avg_results["SID-Recall@1"] == 0.5
    assert query.metric_avg_results["CCE-Recall@1"] == pytest.approx(1 / 3)
    assert user.metric_avg_results["CCE-Recall@1"] == 0.25


def test_only_finite_recall_ndcg_add_extra_views_without_rejecting_other_metrics():
    model = FixedSID()
    metrics = [Recall(1), NDCG(1), Recall(), NDCG(), AUC(), MAP(), MRR()]
    result, _ = evaluation().evaluate(model, metrics, False)
    assert {mt.name for mt in metrics} <= result.metric_avg_results.keys()
    assert {
        name for name in result.metric_avg_results if name.startswith(("SID-", "CCE-"))
    } == {"SID-Recall@1", "SID-NDCG@1", "CCE-Recall@1", "CCE-NDCG@1"}
    assert len(model.calls) == 1
    with warnings.catch_warnings(record=True) as caught:
        evaluation().evaluate(FixedSID(predicted=()), [MRR()], False)
    assert not caught


def test_cutoff_lists_and_flag_off_preserve_legacy_item_metrics():
    ev = evaluation(semantic_id_metrics=False)
    model = FixedSID()
    with pytest.warns(UserWarning, match="disabled"):
        result, _ = ev.evaluate(model, [Recall([1, 2])], False)
    assert not any(
        name.startswith(("SID-", "CCE-")) for name in result.metric_avg_results
    )
    enabled, _ = evaluation().evaluate(FixedSID(), [Recall([1, 2])], False)
    for name in ("Recall@1", "Recall@2"):
        assert enabled.metric_avg_results[name] == result.metric_avg_results[name]
    ordinary, _ = evaluation().evaluate(SPop(), [Recall(1)], False)
    assert not any(
        name.startswith(("SID-", "CCE-")) for name in ordinary.metric_avg_results
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_result_table_union_and_unavailable_values(reverse):
    sid_result, _ = evaluation().evaluate(FixedSID(), [Recall(1)], False)
    ordinary, _ = evaluation().evaluate(SPop(), [Recall(1)], False)
    results = [sid_result, ordinary]
    table = str(ExperimentResult(results[::-1] if reverse else results))
    assert "SID-Recall@1" in table and "CCE-Recall@1" in table
    assert table.count("N/A") == 2
    assert "nan" not in table
    assert "SID-Recall@1" not in ordinary.metric_avg_results


@pytest.mark.parametrize("k", [-1, 1, 2])
def test_combined_ranking_preserves_full_sid_output_and_legacy_scores(k):
    ev = evaluation()
    model = FixedSID(predicted=(9, 0, 0, 1))
    model.fit(ev.train_set)
    ordinary = model.rank(0, [2, 1, 0], k=k, history_items=[0])
    combined = model.rank_with_semantic_ids(0, [2, 1, 0], k=k, history_items=[0])
    for expected, actual in zip(ordinary, combined):
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(combined[2], model.predicted)


def test_rank_fallback_and_unknown_score_padding():
    ev = evaluation()
    model = FixedSID(scores=(2.0, 1.0))
    model.fit(ev.train_set)
    model.num_items = 2
    np.testing.assert_array_equal(
        model.rank_with_semantic_ids(0, [2, 0], history_items=[0])[1], [1.0, 2.0]
    )

    def fail(*args, **kwargs):
        raise ScoreException("no prediction")

    model._score_with_semantic_ids = fail
    ordinary = model.rank(0, history_items=[0])
    combined = model.rank_with_semantic_ids(0, history_items=[0])
    np.testing.assert_array_equal(combined[0], ordinary[0])
    np.testing.assert_array_equal(combined[1], ordinary[1])
    assert combined[2].shape == (0, 1)


def test_lazy_import_without_computing_backends():
    script = """
import sys
class BlockBackends:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'faiss'}:
            raise ImportError('backend unavailable')
sys.meta_path.insert(0, BlockBackends())
from cornac.models import TIGER, LETTER, RPG, DiffGRM, SemanticIDMixin
from cornac.eval_methods import NextItemEvaluation
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


def test_score_fallback_has_no_semantic_credit():
    class Unscorable(FixedSID):
        def _score_with_semantic_ids(self, *args, **kwargs):
            raise ScoreException("no prediction")

    with pytest.warns(UserWarning, match="1 evaluation queries"):
        result, _ = evaluation().evaluate(Unscorable(), [Recall(3)], False)
    assert result.metric_avg_results["Recall@3"] == 1.0
    assert result.metric_avg_results["SID-Recall@3"] == 0.0
    assert result.metric_avg_results["CCE-Recall@3"] == 0.0


def test_unique_complete_ids_make_all_three_views_equal():
    model = FixedSID(codes=(0, 1, 2), predicted=(0, 1, 2))
    result, _ = evaluation().evaluate(model, [Recall(2), NDCG(2)], False)
    for name in ("Recall@2", "NDCG@2"):
        assert (
            result.metric_avg_results[name] == result.metric_avg_results["SID-" + name]
        )
        assert (
            result.metric_avg_results[name] == result.metric_avg_results["CCE-" + name]
        )


def test_empty_output_warns_once_per_split_without_budget_changes():
    train = [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)]
    test = [
        ("u", "te1", "a", 3),
        ("u", "te1", "b", 4),
        ("u", "te2", "a", 5),
        ("u", "te2", "c", 6),
    ]
    ev = NextItemEvaluation.from_splits(
        train, test, fmt="USIT", semantic_id_metrics=True
    )
    model = FixedSID(predicted=())
    with pytest.warns(
        UserWarning, match="2 evaluation queries returned fewer than 2"
    ) as caught:
        result, _ = ev.evaluate(model, [Recall(2), NDCG(2), Recall()], False)
    assert len(caught) == 1
    assert all(
        result.metric_avg_results[name] == 0.0
        for name in ("SID-Recall@2", "CCE-Recall@2", "SID-NDCG@2", "CCE-NDCG@2")
    )
    assert len(model.calls) == 2


def test_exhaustive_sid_order_uses_max_member_and_lexicographic_ties():
    model = FixedSID(codes=(2, 2, 1), scores=(1.0, 3.0, 3.0))
    np.testing.assert_array_equal(
        model._semantic_ids_from_scores(model.scores), [[1], [2]]
    )
