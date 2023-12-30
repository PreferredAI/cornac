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
from cornac.models import GRU4Rec, SPop

buy_data = yoochoose.load_buy(
    reader=Reader(min_sequence_size=2, num_top_freq_item=30000)
)
print("train data loaded")
item_set = set([tup[1] for tup in buy_data])
test_data = yoochoose.load_test(reader=Reader(min_sequence_size=2, item_set=item_set))
print("test data loaded")

next_item_eval = NextItemEvaluation.from_splits(
    train_data=buy_data,
    test_data=test_data[:10000],  # illustration purpose only, subset of test data for faster experiment
    exclude_unknowns=True,
    verbose=True,
    fmt="SITJson",
)

models = [
    SPop(),
    GRU4Rec(
        layers=[100],
        loss="bpr-max",
        n_sample=2048,
        dropout_p_embed=0.0,
        dropout_p_hidden=0.5,
        sample_alpha=0.75,
        batch_size=512,
        n_epochs=10,
        device="cuda",
        verbose=True,
        seed=123,
    )
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
