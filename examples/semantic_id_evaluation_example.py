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

"""Backend-free, synthetic demonstration of shared Semantic ID evaluation."""
import numpy as np

import cornac
from cornac.eval_methods import SemanticIDEvaluation
from cornac.metrics import Recall, NDCG
from cornac.models import SemanticIDCatalog, SemanticIDPrediction, SemanticIDRecommender


class FrozenSIDExample(SemanticIDRecommender):
    """A fixed generator for demonstrating the interface, not a trained model."""

    def __init__(self):
        super().__init__("FrozenSIDExample")

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        codes = np.arange(self.total_items).reshape(-1, 1) // 2
        cardinalities = (int(codes.max()) + 1,)
        self.catalog = SemanticIDCatalog(np.arange(self.total_items), codes,
                                         codes, cardinalities, cardinalities)
        return self

    def semantic_id_catalog(self):
        return self.catalog

    def predict_semantic_ids(self, history_items, **context):
        addresses = np.asarray(sorted(self.catalog.address_to_items), dtype="int64")
        return SemanticIDPrediction(addresses, np.zeros(len(addresses)),
                                    np.ones(len(addresses), dtype=bool))


if __name__ == "__main__":
    train = [("u", "train", "a", 0), ("u", "train", "b", 1),
             ("u", "train", "c", 2)]
    test = [("u", "test", "a", 3), ("u", "test", "b", 4)]
    evaluation = SemanticIDEvaluation.from_splits(
        train, test, fmt="USIT", item_features=np.eye(3),
        feature_source="synthetic identity vectors")
    cornac.Experiment(evaluation, [FrozenSIDExample()], [Recall(2), NDCG(2)],
                      show_validation=False).run()
