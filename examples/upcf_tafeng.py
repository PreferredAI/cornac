# Copyright 2023 The Cornac Authors. All Rights Reserved.
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
"""Example of Recency Aware Collaborative Filtering for Next Basket Recommendation (UPCF)"""

import cornac
from cornac.eval_methods import NextBasketEvaluation
from cornac.metrics import NDCG, HitRatio, Recall
from cornac.models import UPCF

data = cornac.datasets.tafeng.load_basket(
    reader=cornac.data.Reader(
        min_basket_size=3, max_basket_size=50, min_basket_sequence=2
    )
)

next_basket_eval = NextBasketEvaluation(
    data=data, fmt="UBITJson", test_size=0.2, val_size=0.08, seed=123, verbose=True
)

models = [
    UPCF(
        recency=1,
        locality=1,
        asymmetry=0.25,
        verbose=True,
    )
]

metrics = [
    Recall(k=10),
    Recall(k=50),
    NDCG(k=10),
    NDCG(k=50),
    HitRatio(k=10),
    HitRatio(k=50),
]

cornac.Experiment(eval_method=next_basket_eval, models=models, metrics=metrics).run()
