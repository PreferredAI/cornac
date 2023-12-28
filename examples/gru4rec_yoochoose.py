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
"""Example of Session-based Recommendations with Recurrent Neural Networks with Yoochoose data"""

import cornac
from cornac.data import Reader
from cornac.datasets import yoochoose
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import GRU4Rec

buy_data = yoochoose.load_buy(
    reader=Reader(min_sequence_size=3, num_top_freq_item=30000)
)
print("buy data loaded")
test_data = yoochoose.load_test(reader=Reader(min_sequence_size=3))
print("test data loaded")

next_item_eval = NextItemEvaluation.from_splits(
    train_data=buy_data,
    test_data=test_data[:10000],  # illustration purpose only, subset of test data for faster experiment
    verbose=True,
    fmt="SITJson",
)

models = [GRU4Rec()]

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
