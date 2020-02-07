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
"""Your very first example with Cornac"""

import cornac


# Load MovieLens 100K dataset
ml_100k = cornac.datasets.movielens.load_feedback()

# Split data based on ratio
rs = cornac.eval_methods.RatioSplit(
    data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123
)

# Here we are comparing biased MF, PMF, and BPR
mf = cornac.models.MF(
    k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123
)
pmf = cornac.models.PMF(
    k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123
)
bpr = cornac.models.BPR(
    k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123
)

# Define metrics used to evaluate the models
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
recall = cornac.metrics.Recall(k=[10, 20])
ndcg = cornac.metrics.NDCG(k=[10, 20])
auc = cornac.metrics.AUC()

# Put it together into an experiment and run
cornac.Experiment(
    eval_method=rs,
    models=[mf, pmf, bpr],
    metrics=[mae, rmse, recall, ndcg, auc],
    user_based=True,
).run()

