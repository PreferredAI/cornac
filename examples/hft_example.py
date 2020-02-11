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
"""Example for Hidden Factors as Topics (HFT) with Movilen 1M dataset """

import cornac
from cornac.data import Reader
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer


# HFT jointly models the user-item preferences and item texts (e.g., product reviews) with shared item factors
# Below we fit HFT to the MovieLens 1M dataset. We need  both the ratings and movie plots information
plots, movie_ids = movielens.load_plot()
ml_1m = movielens.load_feedback(variant="1M", reader=Reader(item_set=movie_ids))

# Instantiate a TextModality, it makes it convenient to work with text auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
item_text_modality = TextModality(
    corpus=plots,
    ids=movie_ids,
    tokenizer=BaseTokenizer(sep="\t", stop_words="english"),
    max_vocab=5000,
    max_doc_freq=0.5,
)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=ml_1m,
    test_size=0.2,
    exclude_unknowns=True,
    item_text=item_text_modality,
    verbose=True,
    seed=123,
)

# Instantiate HFT model
hft = cornac.models.HFT(
    k=10,
    max_iter=40,
    grad_iter=5,
    l2_reg=0.001,
    lambda_text=0.01,
    vocab_size=5000,
    seed=123,
)

# Instantiate MSE for evaluation
mse = cornac.metrics.MSE()

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split, models=[hft], metrics=[mse], user_based=False
).run()
