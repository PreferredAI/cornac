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


# Load ML-100K dataset
feedback = movielens.load_feedback(variant="100K")

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.2,
    rating_threshold=4.0,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
)

# Comparing a few variants of KNN methods
user_knn_cosine = cornac.models.UserKNN(
    k=20, similarity="cosine", amplify=1.0, name="UserKNN-Cosine"
)
user_knn_pearson = cornac.models.UserKNN(
    k=20, similarity="pearson", amplify=1.0, name="UserKNN-Pearson"
)
user_knn_tfidf = cornac.models.UserKNN(
    k=20, similarity="cosine", weighting="tf-idf", amplify=1.0, name="UserKNN-TFIDF"
)
user_knn_bm25 = cornac.models.UserKNN(
    k=20, similarity="cosine", weighting="bm25", amplify=1.0, name="UserKNN-BM25"
)
item_knn_cosine = cornac.models.ItemKNN(
    k=20, similarity="cosine", amplify=1.0, name="ItemKNN-Cosine"
)
item_knn_adjusted_cosine = cornac.models.ItemKNN(
    k=20, similarity="adjusted", amplify=1.0, name="ItemKNN-AdjustedCosine"
)
item_knn_pearson = cornac.models.ItemKNN(
    k=20, similarity="pearson", amplify=1.0, name="ItemKNN-Pearson"
)

# Evaluation metrics
rmse = cornac.metrics.RMSE()
rec_20 = cornac.metrics.Recall(k=20)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[
        user_knn_cosine,
        user_knn_pearson,
        user_knn_tfidf,
        user_knn_bm25,
        item_knn_cosine,
        item_knn_adjusted_cosine,
        item_knn_pearson,
    ],
    metrics=[rmse, rec_20],
    user_based=True,
).run()
