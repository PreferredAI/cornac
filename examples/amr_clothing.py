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
"""
Example for Adversarial Training Towards Robust Multimedia Recommender System
"""

import cornac
from cornac.datasets import amazon_clothing
from cornac.data import ImageModality
from cornac.eval_methods import RatioSplit


# CausalRec utilises the causal inference to debias the visual bias
# The necessary data can be loaded as follows
feedback = amazon_clothing.load_feedback()
features, item_ids = amazon_clothing.load_visual_feature()  # BIG file

# Instantiate a ImageModality, it makes it convenient to work with visual auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
item_image_modality = ImageModality(features=features, ids=item_ids, normalized=True)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    rating_threshold=0.5,
    exclude_unknowns=True,
    verbose=True,
    item_image=item_image_modality,
)

# Instantiate AMR
amr = cornac.models.AMR(
    k=32,
    k2=32,
    n_epochs=1,
    batch_size=100,
    learning_rate=0.001,
    lambda_w=1,
    lambda_b=0.01,
    lambda_e=0.0,
    lmd=1.0,
    use_gpu=True,
)

# Instantiate evaluation measures
rec_50 = cornac.metrics.Recall(k=50)

# Put everything together into an experiment and run it
cornac.Experiment(eval_method=ratio_split, models=[amr], metrics=[rec_50]).run()
