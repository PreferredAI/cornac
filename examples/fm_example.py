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
"""Example for Factorization Machines with MovieLens 100K dataset"""


import cornac
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import FM


feedback = movielens.load_feedback(variant="100K")

ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    val_size=0.1,
    exclude_unknowns=True,
    verbose=True,
    seed=42,
)

models = [
    FM(k0=1, k1=1, k2=8, method="sgd", max_iter=100, seed=42, name="sgd"),
    FM(k0=1, k1=1, k2=8, method="sgda", max_iter=100, seed=42, name="sgda"),
    FM(k0=1, k1=1, k2=8, method="als", max_iter=100, seed=42, name="als"),
    FM(k0=1, k1=1, k2=8, method="mcmc", max_iter=100, seed=42, name="mcmc"),
]

cornac.Experiment(
    eval_method=ratio_split,
    models=models,
    metrics=[cornac.metrics.RMSE()],
    user_based=False,
).run()
