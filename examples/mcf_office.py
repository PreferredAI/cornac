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
"""Fit to and evaluate MCF on the Office Amazon dataset"""

from cornac.data import GraphModality
from cornac.eval_methods import RatioSplit
from cornac.experiment import Experiment
from cornac import metrics
from cornac.models import MCF
from cornac.datasets import amazon_office as office


# MCF leverages relationships among items, it jointly factorizes the user-item and item-item matrices
# The necessary data can be loaded as follows
ratings = office.load_feedback()
item_net = office.load_graph()

# Instantiate a GraphModality, it makes it convenient to work with graph (network) auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
item_graph_modality = GraphModality(data=item_net)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=ratings,
    test_size=0.2,
    rating_threshold=3.5,
    exclude_unknowns=True,
    verbose=True,
    item_graph=item_graph_modality,
)

# Instantiate MCF model
mcf = MCF(k=10, max_iter=40, learning_rate=0.001, verbose=True)

# Evaluation metrics
ndcg = metrics.NDCG(k=-1)
rmse = metrics.RMSE()
rec = metrics.Recall(k=20)
pre = metrics.Precision(k=20)

# Put everything together into an experiment and run it
Experiment(eval_method=ratio_split, models=[mcf], metrics=[rmse, ndcg, rec, pre]).run()


"""
Output:
    
    |   RMSE | NDCG@-1 | Recall@20 | Precision@20 | Train (s) | Test (s)
--- + ------ + ------- + --------- + ------------ + --------- + --------
MCF | 1.0854 |  0.1598 |    0.0348 |       0.0057 |    7.4057 |   4.1801

*Results may change from one run to another due to different random initial parameters
"""
