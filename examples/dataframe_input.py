# Copyright 2026 The Cornac Authors. All Rights Reserved.
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
"""Example on how to use a pandas DataFrame as input (requires `pip install pandas`)"""

import pandas as pd

import cornac
from cornac.data import Dataset
from cornac.eval_methods import RatioSplit
from cornac.metrics import NDCG, RMSE, Recall
from cornac.models import BPR, MF
from cornac.utils import cache

# Read the MovieLens 100K ratings into a DataFrame
fpath = cache(
    url="https://files.grouplens.org/datasets/movielens/ml-100k/u.data",
    relative_path="ml-100k/u.data",
)
df = pd.read_csv(fpath, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
print(df.head())

# Evaluation methods read the columns of a DataFrame in order: user, item, rating
rs = RatioSplit(
    data=df[["user_id", "item_id", "rating"]],
    test_size=0.2,
    rating_threshold=4.0,
    seed=123,
)

models = [
    MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123),
    BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
]
metrics = [RMSE(), Recall(k=10), NDCG(k=10)]

cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True).run()

# Dataset.from_df reads the columns by name (user_id, item_id and rating by default).
# Train MF on all ratings and recommend 10 movies that user 196 has not rated.
train_set = Dataset.from_df(df, seed=123)
mf = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123)
mf.fit(train_set)
print(mf.recommend(user_id=196, k=10, remove_seen=True, train_set=train_set))
