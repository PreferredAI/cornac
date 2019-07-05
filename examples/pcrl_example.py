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
"""Fit to and evaluate PCRL [1] on the Office Amazon dataset.
[1] Salah, Aghiles, and Hady W. Lauw. Probabilistic Collaborative Representation Learning\
    for Personalized Item Recommendation. In UAI 2018.
"""

from cornac.data import GraphModule
from cornac.eval_methods import RatioSplit
from cornac.experiment import Experiment
from cornac import metrics
from cornac.models import PCRL
from cornac.datasets import amazon_office as office

# Load office ratings and item contexts
ratings = office.load_rating()
contexts = office.load_context()

item_graph_module = GraphModule(data=contexts)

ratio_split = RatioSplit(data=ratings,
                         test_size=0.2, rating_threshold=3.5,
                         shuffle=True, exclude_unknowns=True,
                         verbose=True, item_graph=item_graph_module)

pcrl = PCRL(k=100, z_dims=[300],
            max_iter=300, 
            learning_rate=0.001)


# Evaluation metrics
nDgc = metrics.NDCG(k=-1)
rec = metrics.Recall(k=20)
pre = metrics.Precision(k=20)

# Instantiate and run your experiment
exp = Experiment(eval_method=ratio_split,
                 models=[pcrl],
                 metrics=[nDgc, rec, pre])
exp.run()


"""
Output:
     | NDCG@-1 | Recall@20 | Precision@20 | Train (s) | Test (s)
---- + ------- + --------- + ------------ + --------- + --------
pcrl |  0.1922 |    0.0862 |       0.0148 | 2591.4878 |   4.0957

*Results may change slightly from one run to another due to different random initial parameters
"""