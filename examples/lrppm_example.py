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
"""Example for Learn to Rank user Preferences based on Phrase-level sentiment analysis across Multiple categories (LRPPM)"""

from cornac.datasets import amazon_toy
from cornac.data import SentimentModality
from cornac.eval_methods import RatioSplit
from cornac.metrics import AUC, NDCG, RMSE
from cornac.models import LRPPM
from cornac import Experiment


# Load rating and sentiment information
data = amazon_toy.load_feedback()
sentiment = amazon_toy.load_sentiment()

# Instantiate a SentimentModality, it makes it convenient to work with sentiment information
md = SentimentModality(data=sentiment)

# Define an evaluation method to split feedback into train and test sets
eval_method = RatioSplit(
    data,
    test_size=0.2,
    rating_threshold=1.0,
    sentiment=md,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
)

# Instantiate the model
mter = LRPPM(
    n_factors=8,
    n_ranking_samples=1000,
    n_samples=200,
    ld=1.0,
    reg=0.01,
    max_iter=10000,
    lr=0.1,
    verbose=True,
    seed=123,
)

# Instantiate and run an experiment
Experiment(
    eval_method=eval_method,
    models=[mter],
    metrics=[RMSE(), AUC(), NDCG(k=50)],
).run()
