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

"""Backend-free ordinary item, SID and CCE evaluation in one Experiment."""
import numpy as np

import cornac
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import Recall, NDCG
from cornac.models import NextItemRecommender, SemanticIDMixin, SPop


class FrozenSIDExample(NextItemRecommender, SemanticIDMixin):
    """A fixed generator demonstrating collisions, not a trained model."""

    def __init__(self):
        super().__init__("FrozenSIDExample")

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        self.codes = np.arange(self.total_items).reshape(-1, 1) // 2
        return self

    def semantic_id_catalog(self):
        return self.codes

    def _score_with_semantic_ids(self, user_idx, history_items, **kwargs):
        # a and b share SID 0; this concrete item policy ranks a first.
        scores = -np.arange(self.total_items, dtype="float64")
        predicted = np.unique(self.codes, axis=0)
        return scores, predicted

    def score(self, user_idx, history_items, **kwargs):
        return self._score_with_semantic_ids(user_idx, history_items, **kwargs)[0]


if __name__ == "__main__":
    train = [("u", "train", "a", 0), ("u", "train", "b", 1), ("u", "train", "c", 2)]
    test = [("u", "test", "a", 3), ("u", "test", "b", 4)]
    evaluation = NextItemEvaluation.from_splits(
        train, test, fmt="USIT", semantic_id_metrics=True
    )
    cornac.Experiment(
        evaluation,
        [FrozenSIDExample(), SPop()],
        [Recall(1), NDCG(1)],
        show_validation=False,
        save_dir="/tmp/cornac-semantic-id-example",
    ).run()
