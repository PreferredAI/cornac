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

ml_100k = cn.datasets.movielens.load_100k()
ratio_split = cn.eval_methods.RatioSplit(data=ml_100k, test_size=0.2,
                                         rating_threshold=4.0, verbose=True)

svd = cn.models.SVD(k=10, max_iter=30, learning_rate=0.01, lambda_reg=0.02, verbose=True)

mae = cn.metrics.MAE()
rmse = cn.metrics.RMSE()

cn.Experiment(eval_method=ratio_split,
              models=[svd],
              metrics=[mae, rmse]).run()
