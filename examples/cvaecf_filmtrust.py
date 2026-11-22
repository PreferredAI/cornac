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
"""Fit to and evaluate CVAECF on the FilmTrust dataset"""

from cornac.data import GraphModality
from cornac.eval_methods import RatioSplit
from cornac.experiment import Experiment
from cornac import metrics
from cornac.models import CVAECF
from cornac.datasets import filmtrust

# In addition to learning from preference data, CVAECF further leverages users' auxiliary data (social network in this example).
# The necessary data can be loaded as follows
ratings = filmtrust.load_feedback()
trust = filmtrust.load_trust()

# Instantiate a GraphModality, it makes it convenient to work with graph (network) auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
user_graph_modality = GraphModality(data=trust)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=ratings,
    test_size=0.2,
    rating_threshold=2.5,
    exclude_unknowns=True,
    verbose=True,
    user_graph=user_graph_modality,
    seed=123,
)

# Instantiate CVAECF model
cvaecf = CVAECF(
    z_dim=20,
    h_dim=20,
    autoencoder_structure=[40],
    learning_rate=0.001,
    n_epochs = 70,
    batch_size = 128,
    verbose=True,
    seed = 123
)


# Evaluation metrics
ndcg = metrics.NDCG(k=50)
rec = metrics.Recall(k=50)
pre = metrics.Precision(k=50)

# Put everything together into an experiment and run it
Experiment(
    eval_method=ratio_split,
    models=[cvaecf],
    metrics=[ndcg, pre, rec]
).run()

"""
Output:
       | NDCG@50 | Precision@50 | Recall@50 | Train (s) | Test (s)
------ + ------- + ------------ + --------- + --------- + --------
CVAECF |  0.4171 |       0.0781 |    0.8689 |   13.0752 |   1.4574
"""
