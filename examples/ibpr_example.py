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

import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import IBPR


# Load the MovieLens 1M dataset
ml_1m = movielens.load_feedback(variant="1M")

# Instantiate an evaluation method.
ratio_split = RatioSplit(
    data=ml_1m, test_size=0.2, rating_threshold=1.0, exclude_unknowns=True, verbose=True
)

# Instantiate a IBPR recommender model.
ibpr = IBPR(k=10, verbose=True)

# Instantiate evaluation metrics.
rec_20 = cornac.metrics.Recall(k=20)
pre_20 = cornac.metrics.Precision(k=20)

# Instantiate and then run an experiment.
cornac.Experiment(
    eval_method=ratio_split, models=[ibpr], metrics=[rec_20, pre_20], user_based=True
).run()
