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
"""Example for Matrix Factorization with biases"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit

ratio_split = RatioSplit(data=movielens.load_1m(),
                         test_size=0.2,
                         exclude_unknowns=False,
                         verbose=True)

mf = cornac.models.MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02,
                      use_bias=True, early_stop=True, verbose=True)

mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()

exp = cornac.Experiment(eval_method=ratio_split,
                        models=[mf],
                        metrics=[mae, rmse],
                        user_based=True)
exp.run()
