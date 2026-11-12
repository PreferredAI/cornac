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
"""Example for CrossCBR bundle recommendation on the Youshu dataset."""

import cornac
from cornac.datasets import bundle
from cornac.eval_methods import BundleEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import CrossCBR

# Load the user-bundle splits, bundle membership, user-item feedback and catalogs
train = bundle.load_feedback("Youshu", split="train")
validation = bundle.load_feedback("Youshu", split="validation")
test = bundle.load_feedback("Youshu", split="test")
memberships = bundle.load_bundle_item("Youshu")
user_items = bundle.load_user_item("Youshu")
ids = bundle.load_ids("Youshu")

# Rank all catalog bundles, excluding training positives at test time
evaluation = BundleEvaluation.from_splits(
    train_data=train,
    val_data=validation,
    test_data=test,
    bundle_item_data=memberships,
    user_item_data=user_items,
    test_mask="train",
    exclude_unknowns=True,
    verbose=True,
    **ids,
)

# Instantiate the model and select its best epoch by validation Recall@20
model = CrossCBR(
    emb_size=64,
    num_epochs=100,
    batch_size=2048,
    learning_rate=0.001,
    lambda_cl=0.02,
    early_stopping={"min_delta": 0.0, "patience": 100},
    use_gpu=True,
    seed=2026,
    verbose=True,
)

# Evaluate bundle recommendation through Cornac's experiment pipeline
cornac.Experiment(
    eval_method=evaluation,
    models=[model],
    metrics=[Recall(k=20), NDCG(k=20), Recall(k=40), NDCG(k=40)],
    user_based=True,
).run()
