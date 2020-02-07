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
"""Fit to and evaluate C2PF [1] on the Office Amazon dataset.
[1] Salah, Aghiles, and Hady W. Lauw. A Bayesian Latent Variable Model of User Preferences with Item Context. \
    In IJCAI, pp. 2667-2674. 2018.
"""

from cornac.data import GraphModality
from cornac.eval_methods import RatioSplit
from cornac.experiment import Experiment
from cornac import metrics
from cornac.models import C2PF
from cornac.datasets import amazon_office as office


# In addition to user-item feedback, C2PF integrates item-to-item contextual relationships
# The necessary data can be loaded as follows
ratings = office.load_feedback()
contexts = office.load_graph()

# Instantiate a GraphModality, it makes it convenient to work with graph (network) auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
item_graph_modality = GraphModality(data=contexts)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=ratings,
    test_size=0.2,
    rating_threshold=3.5,
    exclude_unknowns=True,
    verbose=True,
    item_graph=item_graph_modality,
)

# Instantiate C2PF
c2pf = C2PF(k=100, max_iter=80, variant="c2pf")

# Evaluation metrics
ndcg = metrics.NDCG(k=-1)
mrr = metrics.MRR()
rec = metrics.Recall(k=20)
pre = metrics.Precision(k=20)

# Put everything together into an experiment and run it
Experiment(eval_method=ratio_split, models=[c2pf], metrics=[ndcg, mrr, rec, pre]).run()
