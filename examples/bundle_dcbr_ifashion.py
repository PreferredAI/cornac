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
"""Example for DCBR bundle recommendation on iFashion dataset, with GPU required."""

import cornac
from cornac.datasets import bundle
from cornac.eval_methods import BundleEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import DCBR

evaluation = BundleEvaluation.from_splits(
    train_data=bundle.load_feedback("iFashion", split="train"),
    val_data=bundle.load_feedback("iFashion", split="validation"),
    test_data=bundle.load_feedback("iFashion", split="test"),
    bundle_item_data=bundle.load_bundle_item("iFashion"),
    user_item_data=bundle.load_user_item("iFashion"),
    test_mask="train",
    verbose=True,
    **bundle.load_ids("iFashion"),
)

model = DCBR(use_gpu=True, seed=2026, verbose=True)
metrics = [Recall(k=20), NDCG(k=20)]

cornac.Experiment(eval_method=evaluation, models=[model], metrics=metrics).run()
