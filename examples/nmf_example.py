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
"""Example to run Non-negative Matrix Factorization (NMF) model with Ratio Split evaluation strategy"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit

# Load the MovieLens 100K dataset
ml_100k = movielens.load_feedback()

# Instantiate an evaluation method.
eval_method = RatioSplit(
    data=ml_100k,
    test_size=0.2,
    rating_threshold=4.0,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
)

# Instantiate a NMF recommender model.
nmf = cornac.models.NMF(
    k=15,
    max_iter=50,
    learning_rate=0.005,
    lambda_u=0.06,
    lambda_v=0.06,
    lambda_bu=0.02,
    lambda_bi=0.02,
    use_bias=False,
    verbose=True,
    seed=123,
)

# Instantiate evaluation metrics.
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
cornac.Experiment(
    eval_method=eval_method,
    models=[nmf],
    metrics=[mae, rmse, rec_20, pre_20],
    user_based=True,
).run()
