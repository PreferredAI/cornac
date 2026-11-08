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
"""Example for bundle popularity baselines on the Youshu dataset: bundle
recommendation, bundle completion and bundle construction"""

import cornac
from cornac.datasets import bundle
from cornac.eval_methods import BundleConstructionEvaluation, BundleEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import BundlePop

# Load the Youshu user-bundle splits, bundle-item membership, user-item feedback
# and ID catalogs
train = bundle.load_feedback("Youshu", split="train")
validation = bundle.load_feedback("Youshu", split="validation")
test = bundle.load_feedback("Youshu", split="test")
memberships = bundle.load_bundle_item("Youshu")
user_items = bundle.load_user_item("Youshu")
ids = bundle.load_ids("Youshu")

metrics = [Recall(k=20), NDCG(k=20)]

# Bundle recommendation: rank the catalog bundles for each user on the author
# splits, excluding only training positives at test time as CrossCBR does
bundle_eval = BundleEvaluation.from_splits(
    train_data=train,
    val_data=validation,
    test_data=test,
    bundle_item_data=memberships,
    user_item_data=user_items,
    test_mask="train",
    verbose=True,
    **ids,
)
cornac.Experiment(
    eval_method=bundle_eval,
    models=[
        BundlePop(name="BundlePop-bundle", popularity="bundle"),
        BundlePop(name="BundlePop-item", popularity="item"),
        BundlePop(name="BundlePop-user", popularity="user"),
    ],
    metrics=metrics,
).run()

# Bundle completion: split the bundles 70/10/20, give half of the items of each
# held-out bundle and rank its remaining items
completion_eval = BundleConstructionEvaluation(
    data=memberships,
    input_ratio=0.5,
    test_size=0.2,
    val_size=0.1,
    item_ids=ids["constituent_item_ids"],
    seed=123,
    verbose=True,
)
cornac.Experiment(
    eval_method=completion_eval,
    models=[
        BundlePop(name="BundlePop-item", popularity="item"),
        BundlePop(name="BundlePop-cooccurrence", popularity="cooccurrence"),
    ],
    metrics=metrics,
).run()

# Bundle construction: build each held-out bundle for each of its users from
# scratch, with no given items, from the user's bundle and item feedback
construction_eval = BundleConstructionEvaluation(
    data=memberships,
    input_ratio=0,
    test_size=0.2,
    val_size=0.1,
    item_ids=ids["constituent_item_ids"],
    seed=123,
    user_bundle_data=train + validation + test,
    user_item_data=user_items,
    verbose=True,
)
cornac.Experiment(
    eval_method=construction_eval,
    models=[
        BundlePop(name="BundlePop-item", popularity="item"),
        BundlePop(name="BundlePop-bundle", popularity="bundle"),
        BundlePop(name="BundlePop-user", popularity="user"),
    ],
    metrics=metrics,
).run()
