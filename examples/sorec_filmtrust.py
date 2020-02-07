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
"""Fit to and evaluate SoRec on the FilmTrust dataset"""

from cornac.data import GraphModality
from cornac.eval_methods import RatioSplit
from cornac.experiment import Experiment
from cornac import metrics
from cornac.models import SoRec
from cornac.datasets import filmtrust


# SoRec leverages social relationships among users (e.g., trust), it jointly factorizes the user-item and user-user matrices
# The necessary data can be loaded as follows
ratings = filmtrust.load_feedback()
trust = filmtrust.load_trust()

# Instantiate a GraphModality, it makes it convenient to work with graph (network) auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
user_graph_modality = GraphModality(data=trust)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=ratings,
    test_size=0.2,
    rating_threshold=2.5,
    exclude_unknowns=True,
    verbose=True,
    user_graph=user_graph_modality,
    seed=123,
)

# Instantiate SoRec model
sorec = SoRec(k=10, max_iter=50, learning_rate=0.001, verbose=False, seed=123)

# Evaluation metrics
ndcg = metrics.NDCG(k=-1)
rmse = metrics.RMSE()
rec = metrics.Recall(k=20)
pre = metrics.Precision(k=20)

# Put everything together into an experiment and run it
Experiment(
    eval_method=ratio_split, models=[sorec], metrics=[rmse, ndcg, pre, rec]
).run()


"""
Output:
      |   RMSE | NDCG@-1 | Precision@20 | Recall@20 | Train (s) | Test (s)
----- + ------ + ------- + ------------ + --------- + --------- + --------
SoRec | 0.7574 |  0.3707 |       0.0756 |    0.3736 |    0.8198 |   0.8037
"""
