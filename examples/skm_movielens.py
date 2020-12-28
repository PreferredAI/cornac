
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
"""Example SKMeans vs BPR on MovieLens data"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit


# Load user-item feedback
data = movielens.load_feedback(variant="100K")

# Instantiate an evaluation method to split data into train and test sets.
ratio_split = RatioSplit(
    data=data,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=0.5,
)

# Instantiate models
skm = cornac.models.SKMeans(
   k=5,
   max_iter=100,
   tol=1e-10,
   seed=123
)

bpr = cornac.models.BPR(
   k=5,
   max_iter=200,
   learning_rate=0.001,
   lambda_reg=0.01,
   seed=123)


# Instantiate evaluation measures
rec_20 = cornac.metrics.Recall(k=20)
ndcg_20 = cornac.metrics.NDCG(k=20)
auc = cornac.metrics.AUC()



# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[skm, bpr],
    metrics=[rec_20, ndcg_20, auc],
    user_based=True,
).run()