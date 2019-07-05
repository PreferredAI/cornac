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
"""Example for Bayesian Personalized Ranking with Netflix dataset"""

import cornac
from cornac.data import Reader
from cornac.datasets import netflix
from cornac.eval_methods import RatioSplit

ratio_split = RatioSplit(data=netflix.load_data_small(reader=Reader(bin_threshold=1.0)),
                         test_size=0.1, rating_threshold=1.0,
                         exclude_unknowns=True, verbose=True)

bpr = cornac.models.BPR(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.01, verbose=True)

auc = cornac.metrics.AUC()
rec_20 = cornac.metrics.Recall(k=20)

cornac.Experiment(eval_method=ratio_split,
                  models=[bpr],
                  metrics=[auc, rec_20],
                  user_based=True).run()
