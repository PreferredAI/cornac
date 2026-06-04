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
"""Transformer-based next-item recommenders on Diginetica.

SASRec, BERT4Rec, and GPT2Rec share one scoring head (encode the current
session, take the last-position hidden state, dot-product against item
embeddings) and differ only in the sequence encoder:

- SASRec   : its own causal self-attention stack (torch only)
- BERT4Rec : a HuggingFace BERT encoder
- GPT2Rec  : a HuggingFace GPT-2 decoder

BERT4Rec and GPT2Rec require the ``transformers`` package (see each model's
requirements.txt). All three use the next-item-at-last-position objective, not
the canonical MLM/CLM losses in Transformers4Rec paper.
"""

import torch

import cornac
from cornac.datasets import diginetica
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import BERT4Rec, GPT2Rec, GRU4Rec, SASRec

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

train_data = diginetica.load_train()
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

transformer = dict(
    embedding_dim=64,
    loss="cross-entropy",
    n_sample=512,
    batch_size=128,
    n_epochs=100,
    max_len=20,
    num_blocks=2,
    num_heads=2,
    model_selection="best",
    val_eval_every=5,
    val_metric="ndcg",
    val_k=10,
    device=DEVICE,
    verbose=True,
    seed=123,
)

models = [
    GRU4Rec(
        layers=[100],
        loss="cross-entropy",
        dropout_p_hidden=0.3,
        sample_alpha=0.75,
        n_sample=512,
        batch_size=64,
        learning_rate=0.1,
        n_epochs=50,
        model_selection="best",
        val_eval_every=5,
        val_metric="recall",
        val_k=20,
        device=DEVICE,
        verbose=True,
        seed=123,
    ),
    SASRec(learning_rate=0.01, **transformer),
    BERT4Rec(learning_rate=0.01, **transformer),
    GPT2Rec(learning_rate=0.001, **transformer),
]

metrics = [
    NDCG(k=10),
    NDCG(k=50),
    Recall(k=10),
    Recall(k=50),
    MRR(),
]

cornac.Experiment(
    eval_method=next_item_eval,
    models=models,
    metrics=metrics,
).run()
