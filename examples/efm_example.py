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
"""Example for Explicit Factor Models"""

import cornac
from cornac.datasets import amazon_toy
from cornac.data import SentimentModality
from cornac.eval_methods import RatioSplit


# Load rating and sentiment information
rating = amazon_toy.load_feedback()
sentiment = amazon_toy.load_sentiment()

# Instantiate a SentimentModality, it makes it convenient to work with sentiment information
md = SentimentModality(data=sentiment)

# Define an evaluation method to split feedback into train and test sets
split_data = RatioSplit(
    data=rating,
    test_size=0.15,
    exclude_unknowns=True,
    verbose=True,
    sentiment=md,
    seed=123,
)

# Instantiate the EFM model
efm = cornac.models.EFM(
    num_explicit_factors=40,
    num_latent_factors=60,
    num_most_cared_aspects=15,
    rating_scale=5.0,
    alpha=0.85,
    lambda_x=1,
    lambda_y=1,
    lambda_u=0.01,
    lambda_h=0.01,
    lambda_v=0.01,
    max_iter=100,
    num_threads=1,
    trainable=True,
    verbose=True,
    seed=123,
)

# Instantiate evaluation metrics
rmse = cornac.metrics.RMSE()
ndcg_50 = cornac.metrics.NDCG(k=50)
auc = cornac.metrics.AUC()

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=split_data, models=[efm], metrics=[rmse, ndcg_50, auc]
).run()
