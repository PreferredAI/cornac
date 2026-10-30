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
"""RPG (parallel long semantic IDs) on Diginetica.

RPG quantizes item CONTENT embeddings into long *unordered* semantic IDs with
product quantization (OPQ, via faiss), then trains a GPT-2 decoder that predicts
every digit of the next item's semantic ID in parallel (multi-token prediction).
Inference uses a similarity graph over item semantic IDs to guide beam decoding.

The content embeddings are precomputed and passed to the evaluation method as a
FeatureModality. Diginetica ships without item text in Cornac, so this example
uses random vectors as a stand-in -- replace them with real content embeddings
for meaningful semantic IDs, e.g. with the OpenAI ``text-embedding-3-large``
embeddings used in the paper, or with sentence-transformers::

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-t5-base")
    features = encoder.encode([titles[iid] for iid in item_ids])

Requires ``faiss-cpu`` and ``scikit-learn`` (see cornac/models/rpg/requirements.txt).
"""

import numpy as np
import torch

import cornac
from cornac.data import FeatureModality
from cornac.datasets import diginetica
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import RPG, SASRec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

train_data = diginetica.load_train()
val_data = diginetica.load_val()
test_data = diginetica.load_test()
print("data loaded")

item_ids = sorted({tup[2] for tup in train_data + val_data + test_data})
print("NOTE: using random item features as a stand-in; " "replace with real content embeddings (see module docstring).")
features = np.random.RandomState(123).randn(len(item_ids), 768).astype("float32")

next_item_eval = NextItemEvaluation.from_splits(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    exclude_unknowns=True,
    semantic_id_metrics=True,
    verbose=True,
    fmt="USIT",
    item_feature=FeatureModality(features=features, ids=item_ids),
)

models = [
    RPG(
        n_codebook=32,  # semantic-ID length; set to 64 for the long-ID config
        codebook_size=256,
        pca_dim=512,
        n_epochs=50,
        batch_size=256,
        max_len=50,
        scoring="graph",  # or "exact" for the full-catalog parallel-MTP ranking
        n_beams=50,
        graph_edges=50,
        propagation_steps=3,
        device=DEVICE,
        verbose=True,
        seed=2024,
    ),
    SASRec(
        embedding_dim=64,
        loss="cross-entropy",
        n_sample=512,
        batch_size=128,
        n_epochs=100,
        max_len=20,
        num_blocks=2,
        num_heads=2,
        learning_rate=0.01,
        device=DEVICE,
        verbose=True,
        seed=123,
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
