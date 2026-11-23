# Copyright 2018 The Cornac Authors. All Rights Reserved.
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

from unittest.mock import patch

import pytest

from cornac.eval_methods import BaseMethod
from cornac.metrics import Recall
from cornac.models import MostPop


TRAIN = [("u", "a", 4.0), ("u", "b", 3.0), ("v", "a", 5.0), ("v", "c", 4.0)]
TEST = [("u", "c", 5.0), ("v", "b", 5.0)]


@pytest.mark.parametrize("user_based", [False, True])
def test_validation_only_item_outside_test_mask(user_based):
    split = BaseMethod.from_splits(
        TRAIN, TEST, val_data=[("u", "val-only", 3.0)], exclude_unknowns=False, seed=12
    )
    val_item = split.global_iid_map["val-only"]
    assert val_item == split.test_set.num_items == 3
    assert split.val_set.num_items == 4

    model = MostPop()
    with patch.object(model, "rank", wraps=model.rank) as rank:
        test_result, _ = split.evaluate(
            model, [Recall(k=3)], user_based=user_based, show_validation=False
        )

    candidates = {
        call.kwargs["user_idx"]: call.kwargs["item_indices"].tolist()
        for call in rank.call_args_list
    }
    # Each user has one held-out positive. Their other two items are in training.
    assert candidates == {0: [2], 1: [1]}
    assert test_result.metric_user_results["Recall@3"] == {0: 1.0, 1: 1.0}
    assert test_result.metric_avg_results["Recall@3"] == 1.0


@pytest.mark.parametrize(
    "val_data, exclude_unknowns, expected_candidates",
    [
        pytest.param(None, False, [2, 3], id="no-validation"),
        pytest.param([("u", "d", 3.0)], False, [2], id="in-mask-validation"),
        pytest.param(
            [("u", "d", 3.0), ("u", "val-only", 3.0)],
            True,
            [2],
            id="exclude-unknowns",
        ),
    ],
)
def test_training_known_validation_candidates(
    val_data, exclude_unknowns, expected_candidates
):
    # d is known to training through v, and is otherwise a negative for u.
    split = BaseMethod.from_splits(
        TRAIN + [("v", "d", 3.0)],
        TEST,
        val_data=val_data,
        exclude_unknowns=exclude_unknowns,
        seed=12,
    )
    assert split.train_set.num_items == split.test_set.num_items == 4
    if exclude_unknowns:
        assert "val-only" not in split.global_iid_map

    model = MostPop()
    with patch.object(model, "rank", wraps=model.rank) as rank:
        result, _ = split.evaluate(
            model, [Recall(k=3)], user_based=True, show_validation=False
        )

    candidates = {
        call.kwargs["user_idx"]: call.kwargs["item_indices"].tolist()
        for call in rank.call_args_list
    }
    assert candidates == {0: expected_candidates, 1: [1]}
    assert result.metric_user_results["Recall@3"] == {0: 1.0, 1: 1.0}


@pytest.mark.parametrize("with_val_only", [False, True])
def test_validation_positive_for_test_only_item_inside_mask(with_val_only):
    # cold is not in training. v puts it in the test universe, so u's
    # validation interaction must still exclude it from u's negatives.
    val_data = [("u", "cold", 5.0)]
    if with_val_only:
        val_data.append(("u", "val-only", 5.0))
    split = BaseMethod.from_splits(
        TRAIN,
        [("u", "c", 5.0), ("v", "cold", 5.0)],
        val_data=val_data,
        exclude_unknowns=False,
        seed=12,
    )
    cold_item = split.global_iid_map["cold"]
    assert cold_item == split.train_set.num_items < split.test_set.num_items
    if with_val_only:
        assert split.global_iid_map["val-only"] == split.test_set.num_items

    model = MostPop()
    with patch.object(model, "rank", wraps=model.rank) as rank:
        result, _ = split.evaluate(
            model, [Recall(k=3)], user_based=True, show_validation=False
        )

    candidates = {
        call.kwargs["user_idx"]: call.kwargs["item_indices"].tolist()
        for call in rank.call_args_list
    }
    assert candidates == {0: [2], 1: [1, cold_item]}
    assert result.metric_user_results["Recall@3"] == {0: 1.0, 1: 1.0}
