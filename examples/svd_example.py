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


import cornac as cn


# Load the MovieLens 100K dataset
ml_100k = cn.datasets.movielens.load_feedback()

# Instantiate an evaluation method to split data into train and test sets.
ratio_split = cn.eval_methods.RatioSplit(
    data=ml_100k, test_size=0.2, rating_threshold=4.0, verbose=True
)

# Instantiate the models of interest
bo = cn.models.BaselineOnly(
    max_iter=30, learning_rate=0.01, lambda_reg=0.02, verbose=True
)
svd = cn.models.SVD(
    k=10, max_iter=30, learning_rate=0.01, lambda_reg=0.02, verbose=True
)

# Instantiate evaluation measures
mae = cn.metrics.MAE()
rmse = cn.metrics.RMSE()

# Instantiate and run an experiment.
cn.Experiment(eval_method=ratio_split, models=[bo, svd], metrics=[mae, rmse]).run()
