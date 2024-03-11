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
"""Example for comparing different ANN Searchers with BPR model"""

import cornac
from cornac.data import Reader
from cornac.datasets.netflix import load_feedback
from cornac.eval_methods import RatioSplit
from cornac.metrics import AUC, Recall
from cornac.models import BPR, AnnoyANN, FaissANN, HNSWLibANN, ScaNNANN


bpr = BPR(k=50, max_iter=200, learning_rate=0.001, lambda_reg=0.001, verbose=True)

# using default params of the ANN searchers
# performance could be better if they are carefuly tuned
ann1 = AnnoyANN(bpr, verbose=True)
ann2 = FaissANN(bpr, verbose=True)
ann3 = HNSWLibANN(bpr, verbose=True)
ann4 = ScaNNANN(bpr, verbose=True)

cornac.Experiment(
    eval_method=RatioSplit(
        data=load_feedback(variant="small", reader=Reader(bin_threshold=1.0)),
        test_size=0.1,
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=True,
    ),
    models=[bpr, ann1, ann2, ann3, ann4],
    metrics=[AUC(), Recall(k=50)],
    user_based=True,
).run()
