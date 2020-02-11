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
"""Example for Maximum Margin Matrix Factorization on MovieLens 100K dataset"""

import cornac


# Load MovieLens 100K dataset, and binarise ratings using cornac.data.Reader
feedback = cornac.datasets.movielens.load_feedback(
    variant="100K", reader=cornac.data.Reader(bin_threshold=1.0)
)

# Define an evaluation method to split feedback into train and test sets
ratio_split = cornac.eval_methods.RatioSplit(data=feedback, test_size=0.2, verbose=True)

# Instantiate MMMF model
mmmf = cornac.models.MMMF(k=10, max_iter=200, learning_rate=0.01, verbose=True)

# Use NDCG@10 for evaluation
ndcg = cornac.metrics.NDCG(k=10)

# Put everything together into an experiment and run it
cornac.Experiment(eval_method=ratio_split, models=[mmmf], metrics=[ndcg]).run()
