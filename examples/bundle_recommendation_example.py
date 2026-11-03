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

"""Recommend existing catalog bundles with ordinary BPR.

Bundle construction remains future work.
Run from the repository root: python examples/bundle_recommendation_example.py
"""

import cornac
from cornac.eval_methods import BundleEvaluation
from cornac.models import BPR


if __name__ == "__main__":
    # Bundle IDs and constituent item IDs occupy separate namespaces.
    catalog = [("starter", "pen"), ("starter", "paper"),
               ("artist", "paper"), ("artist", "paint"),
               ("writer", "pen"), ("writer", "notebook")]
    training = [("alice", "starter", 1), ("bob", "artist", 1)]
    testing = [("alice", "writer", 1), ("bob", "starter", 1)]
    # Optional: independently observed item feedback from training time only.
    item_feedback = [("alice", "notebook", 1), ("bob", "paint", 1)]
    evaluation = BundleEvaluation.from_splits(
        train_data=training, test_data=testing, bundle_item_data=catalog,
        user_item_data=item_feedback, exclude_unknowns=True, seed=42,
    )
    model = BPR(k=4, max_iter=10, seed=42)
    cornac.Experiment(
        eval_method=evaluation, models=[model],
        metrics=[cornac.metrics.Recall(k=2), cornac.metrics.NDCG(k=2)],
        user_based=True,
    ).run()
    # BPR uses user-bundle feedback. Specialized bundle models additionally use
    # train_set.bundle_item_matrix and user_item_matrix.
    print(model.recommend("alice", k=2, remove_seen=True,
                          train_set=evaluation.train_set))
