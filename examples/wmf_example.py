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
"""Example for Collaborative Filtering for Implicit Feedback Datasets (Citeulike)"""

import cornac
from cornac.data import Reader
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit

# Load user-item feedback
_, item_ids = citeulike.load_text()
data = citeulike.load_feedback(reader=Reader(item_set=item_ids))

# Instantiate an evaluation method to split data into train and test sets.
ratio_split = RatioSplit(
    data=data,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=0.5,
)

# Instantiate the WMF model
wmf = cornac.models.WMF(
    k=50,
    max_iter=50,
    learning_rate=0.001,
    lambda_u=0.01,
    lambda_v=0.01,
    verbose=True,
    seed=123,
)

# Use Recall@300 for evaluation
rec_300 = cornac.metrics.Recall(k=300)

# Instantiate and run an experiment
cornac.Experiment(
    eval_method=ratio_split, models=[wmf], metrics=[rec_300], user_based=True
).run()
