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
"""Export the results of an experiment to a pandas DataFrame and a JSON file
(requires `pip install pandas` for the DataFrame)"""

import cornac
from cornac.eval_methods import RatioSplit
from cornac.metrics import MAE, NDCG, RMSE, Recall
from cornac.models import BPR, MF, PMF

# Load MovieLens 100K and split it into training, validation and test sets
ml_100k = cornac.datasets.movielens.load_feedback()
rs = RatioSplit(
    data=ml_100k, test_size=0.2, val_size=0.1, rating_threshold=4.0, seed=123
)

# Compare biased MF, PMF and BPR
models = [
    MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=123),
    PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
    BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
]

# Rating and ranking metrics
metrics = [MAE(), RMSE(), Recall(k=10), NDCG(k=10)]

# Keep the experiment to use its results after it runs
exp = cornac.Experiment(eval_method=rs, models=models, metrics=metrics)
exp.run()

# Test results as a DataFrame: one row per model, one column per metric
print(exp.result.to_dataframe())

# Model parameters, evaluation method and seed, and test results of each model
exp.result.to_json("export_results.json")
