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
"""Example (EASEᴿ) Embarrassingly Shallow Autoencoders for Sparse Data on MovieLens data"""

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit


# Load user-item feedback
data = movielens.load_feedback(variant="1M")

# Instantiate an evaluation method to split data into train and test sets.
ratio_split = RatioSplit(
    data=data,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=0.8,
)

ease_original = cornac.models.EASE(
    lamb=500,
    name="EASEᴿ (B>0)",
    posB=True
)

ease_all = cornac.models.EASE(
    lamb=500,
    name="EASEᴿ (B>-∞)",
    posB=False
)


# Instantiate evaluation measures
rec_20 = cornac.metrics.Recall(k=20)
rec_50 = cornac.metrics.Recall(k=50)
ndcg_100 = cornac.metrics.NDCG(k=100)


# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[ease_original, ease_all],
    metrics=[rec_20, rec_50, ndcg_100],
    user_based=True, #If `False`, results will be averaged over the number of ratings.
    save_dir=None
).run()