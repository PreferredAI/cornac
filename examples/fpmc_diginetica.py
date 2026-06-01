# Copyright 2026 The Cornac Authors. All Rights Reserved.
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
"""Example of Factorizing Personalized Markov Chains (FPMC) with Diginetica data"""

import cornac
from cornac.datasets import diginetica
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import FPMC

train_data = diginetica.load_train()
# load_val/load_test default to mode="session-based": each user's single
# held-out session, so the model is never scored on transitions it trained on.
val_data = diginetica.load_val()
test_data = diginetica.load_test()
print("data loaded")

next_item_eval = NextItemEvaluation.from_splits(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    exclude_unknowns=True,
    verbose=True,
    fmt="USIT",
)

model = FPMC(
    embedding_dim=64,
    loss="cross-entropy",
    n_sample=512,
    batch_size=128,
    learning_rate=0.1,
    n_epochs=100,
    model_selection="best",
    val_eval_every=5,
    val_metric="ndcg",
    val_k=10,
    device="cpu",
    verbose=True,
    seed=123,
)

metrics = [
    NDCG(k=10),
    NDCG(k=50),
    Recall(k=10),
    Recall(k=50),
    MRR(),
]

cornac.Experiment(
    eval_method=next_item_eval,
    models=[model],
    metrics=metrics,
).run()
