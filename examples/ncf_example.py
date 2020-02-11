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
from cornac.eval_methods import RatioSplit
from cornac.datasets import amazon_clothing
from cornac.data import Reader


# Load the Amazon Clothing  dataset, and binarise ratings using cornac.data.Reader
feedback = amazon_clothing.load_feedback(reader=Reader(bin_threshold=1.0))

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.2,
    rating_threshold=1.0,
    seed=123,
    exclude_unknowns=True,
    verbose=True,
)

# Instantiate the recommender models to be compared
gmf = cornac.models.GMF(
    num_factors=8,
    num_epochs=10,
    learner="adam",
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
)
mlp = cornac.models.MLP(
    layers=[64, 32, 16, 8],
    act_fn="tanh",
    learner="adam",
    num_epochs=10,
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
)
neumf1 = cornac.models.NeuMF(
    num_factors=8,
    layers=[64, 32, 16, 8],
    act_fn="tanh",
    learner="adam",
    num_epochs=10,
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
)
neumf2 = cornac.models.NeuMF(
    name="NeuMF_pretrained",
    learner="adam",
    num_epochs=10,
    batch_size=256,
    lr=0.001,
    num_neg=50,
    seed=123,
    num_factors=gmf.num_factors,
    layers=mlp.layers,
    act_fn=mlp.act_fn,
).pretrain(gmf, mlp)

# Instantiate evaluation metrics
ndcg_50 = cornac.metrics.NDCG(k=50)
rec_50 = cornac.metrics.Recall(k=50)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[gmf, mlp, neumf1, neumf2],
    metrics=[ndcg_50, rec_50],
).run()
