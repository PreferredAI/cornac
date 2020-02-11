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
"""Example for hyper-parameter searching with Matrix Factorization"""

import numpy as np
import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.hyperopt import Discrete, Continuous
from cornac.hyperopt import GridSearch, RandomSearch


# Load MovieLens 100K ratings
ml_100k = movielens.load_feedback(variant="100K")

# Define an evaluation method to split feedback into train, validation and test sets
ratio_split = RatioSplit(data=ml_100k, test_size=0.1, val_size=0.1, verbose=True)

# Instantiate MAE and RMSE for evaluation
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()

# Define a base MF model with fixed hyper-parameters
mf = cornac.models.MF(max_iter=20, learning_rate=0.01, early_stop=True, verbose=True)

# Wrap MF model inside GridSearch along with the searching space
gs_mf = GridSearch(
    model=mf,
    space=[
        Discrete("k", [10, 30, 50]),
        Discrete("use_bias", [True, False]),
        Discrete("lambda_reg", [1e-1, 1e-2, 1e-3, 1e-4]),
    ],
    metric=rmse,
    eval_method=ratio_split,
)

# Wrap MF model inside RandomSearch along with the searching space, try 30 times
rs_mf = RandomSearch(
    model=mf,
    space=[
        Discrete("k", [10, 30, 50]),
        Discrete("use_bias", [True, False]),
        Continuous("lambda_reg", low=1e-4, high=1e-1),
    ],
    metric=rmse,
    eval_method=ratio_split,
    n_trails=30,
)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[gs_mf, rs_mf],
    metrics=[mae, rmse],
    user_based=False,
).run()
