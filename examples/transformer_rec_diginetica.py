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

TransformerRec is one model with two axes of configuration: a HuggingFace
backbone (bert/gpt2/xlnet/electra) and a language-modeling objective
(clm/mlm/plm/rtd). This example compares:

- SASRec                          : causal self-attention baseline (torch only)
- TransformerRec gpt2+clm (all)   : causal LM, loss at every position
- TransformerRec gpt2+clm (last)  : legacy prefix breakdown, loss at last
                                    position only (the old GPT2Rec behavior)
- TransformerRec bert+mlm         : BERT4Rec-style Cloze training

TransformerRec requires the ``transformers`` package (see the model's
requirements.txt). The clm-all vs clm-last pair isolates the effect of the
training setting under identical architecture and hyperparameters.
"""

import torch

import cornac
from cornac.datasets import diginetica
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import SASRec, TransformerRec

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

shared = dict(
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
    SASRec(learning_rate=0.01, **shared),
    TransformerRec(
        name="TransformerRec-gpt2-clm-all",
        backbone="gpt2",
        objective="clm",
        loss_at="all",
        learning_rate=0.001,
        **shared,
    ),
    TransformerRec(
        name="TransformerRec-gpt2-clm-last",
        backbone="gpt2",
        objective="clm",
        loss_at="last",
        learning_rate=0.001,
        **shared,
    ),
    TransformerRec(
        name="TransformerRec-bert-mlm",
        backbone="bert",
        objective="mlm",
        learning_rate=0.01,
        **shared,
    ),
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
