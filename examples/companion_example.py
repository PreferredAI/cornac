# Copyright 2024 The Cornac Authors. All Rights Reserved.
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
"""Example for Comparative Aspects and Opinions Ranking for Recommendation Explanations"""

import cornac
from cornac.datasets import amazon_toy
from cornac.data import SentimentModality
from cornac.eval_methods import StratifiedSplit
from cornac.metrics import NDCG, RMSE, AUC
from cornac import Experiment


rating = amazon_toy.load_feedback(fmt="UIRT")
sentiment = amazon_toy.load_sentiment()


# Instantiate a SentimentModality, it makes it convenient to work with sentiment information
md = SentimentModality(data=sentiment)

# Define an evaluation method to split feedback into train and test sets
eval_method = StratifiedSplit(
    rating,
    group_by="user",
    chrono=True,
    sentiment=md,
    test_size=1,
    val_size=1,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
)

companion = cornac.models.Companion(
    n_top_aspects=0,
    max_iter=10000,
    verbose=True,
    seed=123,
)

# Instantiate and run an experiment
exp = Experiment(
    eval_method=eval_method,
    models=[companion],
    metrics=[RMSE(), NDCG(k=20), AUC()],
)
exp.run()
