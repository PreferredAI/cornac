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
"""Example for Nearest Neighborhood-based methods with MovieLens 100K dataset"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit


K = 50  # number of nearest neighbors


# Load ML-100K dataset
feedback = movielens.load_feedback(variant="100K")

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback, test_size=0.2, exclude_unknowns=True, verbose=True, seed=123
)

# UserKNN methods
user_knn_cosine = cornac.models.UserKNN(k=K, similarity="cosine", name="UserKNN-Cosine")
user_knn_pearson = cornac.models.UserKNN(
    k=K, similarity="pearson", name="UserKNN-Pearson"
)
user_knn_amp = cornac.models.UserKNN(
    k=K, similarity="cosine", amplify=2.0, name="UserKNN-Amplified"
)
user_knn_idf = cornac.models.UserKNN(
    k=K, similarity="cosine", weighting="idf", name="UserKNN-IDF"
)
user_knn_bm25 = cornac.models.UserKNN(
    k=K, similarity="cosine", weighting="bm25", name="UserKNN-BM25"
)
# ItemKNN methods
item_knn_cosine = cornac.models.ItemKNN(k=K, similarity="cosine", name="ItemKNN-Cosine")
item_knn_pearson = cornac.models.ItemKNN(
    k=K, similarity="pearson", name="ItemKNN-Pearson"
)
item_knn_adjusted = cornac.models.ItemKNN(
    k=K, similarity="cosine", mean_centered=True, name="ItemKNN-AdjustedCosine"
)

# Put everything together into an experiment
cornac.Experiment(
    eval_method=ratio_split,
    models=[
        user_knn_cosine,
        user_knn_pearson,
        user_knn_amp,
        user_knn_idf,
        user_knn_bm25,
        item_knn_cosine,
        item_knn_pearson,
        item_knn_adjusted,
    ],
    metrics=[cornac.metrics.RMSE()],
    user_based=True,
).run()
