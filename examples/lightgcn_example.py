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
"""
Example for LightGCN, using the CiteULike dataset
"""
import cornac
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit

# Load user-item feedback
data = citeulike.load_feedback()

# Instantiate an evaluation method to split data into train and test sets.
ratio_split = RatioSplit(
    data=data,
    val_size=0.1,
    test_size=0.1,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=0.5,
)

# Instantiate the LightGCN model
lightgcn = cornac.models.LightGCN(
    seed=123,
    num_epochs=1000,
    num_layers=3,
    early_stopping={"min_delta": 1e-4, "patience": 50},
    batch_size=1024,
    learning_rate=0.001,
    lambda_reg=1e-4,
    verbose=True
)

# Instantiate evaluation measures
rec_20 = cornac.metrics.Recall(k=20)
ndcg_20 = cornac.metrics.NDCG(k=20)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[lightgcn],
    metrics=[rec_20, ndcg_20],
    user_based=True,
).run()
