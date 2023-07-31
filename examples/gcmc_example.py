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
Example for Graph Convolutional Matrix Completion with MovieLens 100K dataset
"""
import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit

# Load user-item feedback
data_100k = movielens.load_feedback(variant="100K")

# Instantiate an evaluation method to split data into train and test sets.
ratio_split = RatioSplit(
    data=data_100k,
    test_size=0.1,
    val_size=0.1,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
)

pmf = cornac.models.PMF(
    k=10,
    max_iter=100,
    learning_rate=0.001,
    lambda_reg=0.001
)

biased_mf = cornac.models.MF(
    name="BiasMF",
    k=10,
    max_iter=25,
    learning_rate=0.01,
    lambda_reg=0.02,
    use_bias=True,
    seed=123
)

gcmc = cornac.models.GCMC(
    seed=123,
)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[pmf, biased_mf, gcmc],
    metrics=[cornac.metrics.RMSE()],
    user_based=False,
).run()
