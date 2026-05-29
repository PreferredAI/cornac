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
"""Sequential recommendation on Diginetica with GRU4Rec / SASRec / BERT4Rec / GPT2Rec.

Each model is trained twice:

- ``mode="sbr"``  (session-based: only the current session is context)
- ``mode="sar"``  (session-aware:  the user's cross-session history is context)

``"sbr"`` and ``"sar"`` are short-form aliases for ``"session-based"`` and
``"session-aware"`` and are matched case-insensitively.

Runs a short 10-epoch schedule so the example fits in a reasonable time;
bump ``N_EPOCHS`` for real comparisons.

Note: GRU4Rec converges slower than Transformers-based models.
"""

import cornac
from cornac.data import Reader
from cornac.datasets import diginetica
from cornac.eval_methods import SequentialEvaluation
from cornac.metrics import AUC, MRR, NDCG, Recall
from cornac.models import BERT4Rec, GPT2Rec, GRU4Rec, SASRec

N_EPOCHS = 10
DEVICE = "cuda"  # set to "cpu" if no GPU is available
SEED = 123

reader = Reader(min_sequence_size=2)
train_data = diginetica.load_train(reader=reader)
val_data = diginetica.load_val(reader=reader)
test_data = diginetica.load_test(reader=reader)

eval_method = SequentialEvaluation.from_splits(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    fmt="USIT",
    exclude_unknowns=True,
    verbose=True,
    mode="last",
)

shared = dict(
    embedding_dim=64,
    batch_size=256,
    n_sample=512,
    n_epochs=N_EPOCHS,
    max_len=20,
    num_blocks=2,
    num_heads=2,
    device=DEVICE,
    verbose=True,
    seed=SEED,
)

models = []
for setting in ("sbr", "sar"):
    models += [
        GRU4Rec(
            name=f"GRU4Rec-{setting}",
            layers=[64],
            loss="cross-entropy",
            mode=setting,
            batch_size=shared["batch_size"],
            n_sample=shared["n_sample"],
            sample_alpha=0.75,
            dropout_p_hidden=0.3,
            n_epochs=N_EPOCHS,
            device=DEVICE,
            verbose=True,
            seed=SEED,
        ),
        SASRec(name=f"SASRec-{setting}", mode=setting, **shared),
        BERT4Rec(name=f"BERT4Rec-{setting}", mode=setting, **shared),
        GPT2Rec(name=f"GPT2Rec-{setting}", mode=setting, **shared),
    ]

metrics = [AUC(), MRR(), NDCG(k=10), NDCG(k=50), Recall(k=10), Recall(k=50)]

cornac.Experiment(
    eval_method=eval_method,
    models=models,
    metrics=metrics,
).run()
