# Copyright 2023 The Cornac Authors. All Rights Reserved.
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
"""Example of a next-item recommendation model based on item popularity"""

import cornac
from cornac.datasets import yoochoose
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import SPop

buy_data = yoochoose.load_buy()
print("buy data loaded")
test_data = yoochoose.load_test()
print("test data loaded")

next_item_eval = NextItemEvaluation.from_splits(
    train_data=buy_data,
    test_data=test_data[:10000],  # illustration purpose only, subset of test data for faster experiment
    verbose=True,
    fmt="SITJson",
)

models = [
    SPop(name="Pop", use_session_popularity=False),
    SPop(),
]

metrics = [
    NDCG(k=10),
    NDCG(k=50),
    Recall(k=10),
    Recall(k=50),
    MRR(),
]

cornac.Experiment(
    eval_method=next_item_eval,
    models=models,
    metrics=metrics,
).run()
