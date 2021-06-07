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
Propensity-based Stratified Evaluation Method

Reference:
---------
Amir H. Jadidinejad, Craig Macdonald and Iadh Ounis, 
The Simpson's Paradox in the Offline Evaluation of Recommendation Systems, 
ACM Transactions on Information Systems (to appear)
https://arxiv.org/abs/2104.08912
"""
import cornac
from cornac.models import WMF, BPR
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

from cornac.eval_methods import PropensityStratifiedEvaluation
from cornac.experiment import Experiment

# Load the MovieLens 1M dataset
ml_dataset = cornac.datasets.movielens.load_feedback(variant="1M")

# Instantiate an instance of PropensityStratifiedEvaluation method
stra_eval_method = PropensityStratifiedEvaluation(data=ml_dataset,
                                                  n_strata=2,  # number of strata
                                                  rating_threshold=4.0,
                                                  verbose=True)

# define the examined models
models = [
    WMF(k=10, seed=123),
    BPR(k=10, seed=123),
]

# define the metrics
metrics = [MAE(), RMSE(), Precision(k=10),
           Recall(k=10), NDCG(), AUC(), MAP()]

# run an experiment
exp_stra = Experiment(eval_method=stra_eval_method,
                      models=models, metrics=metrics)

exp_stra.run()
