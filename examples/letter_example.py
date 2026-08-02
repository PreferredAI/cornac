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
"""LETTER (learnable tokenizer for generative retrieval) on Diginetica.

LETTER replaces TIGER's RQ-VAE tokenizer with the released LETTER tokenizer:
(1) a collaborative InfoNCE loss aligning each item's semantic ID with a
precomputed collaborative (CF) item embedding, and (2) a diversity loss that
spreads codebook usage. Its downstream generator follows the released tied
T5-vocabulary/EOS objective and epoch-level validation-loss early stopping.

Two things are precomputed and passed in:
  * item CONTENT embeddings -> the evaluation method's FeatureModality, e.g.
    with sentence-transformers::

        from sentence_transformers import SentenceTransformer
        content = SentenceTransformer("sentence-t5-base").encode(titles)

  * item COLLABORATIVE embeddings -> ``LETTER(cf_embeddings=...,
    cf_embedding_ids=...)``, typically the item embeddings of a trained CF
    model (SASRec in the paper). Raw IDs are supplied so LETTER can align the
    rows to Cornac's global item indices.

Diginetica ships without item text/CF vectors in Cornac, so this example uses
random vectors as stand-ins -- replace both with real embeddings for
meaningful semantic IDs.
"""

import numpy as np
import torch

import cornac
from cornac.data import FeatureModality
from cornac.datasets import diginetica
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import LETTER, TIGER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

train_data = diginetica.load_train()
val_data = diginetica.load_val()
test_data = diginetica.load_test()
print("data loaded")

item_ids = sorted({tup[2] for tup in train_data + val_data + test_data})
rng = np.random.RandomState(123)
print(
    "NOTE: using random content + CF features as stand-ins; replace with real "
    "content embeddings and trained-CF item embeddings (see module docstring)."
)
content = rng.randn(len(item_ids), 768).astype("float32")
cf_embeddings = rng.randn(len(item_ids), 32).astype("float32")  # e.g. SASRec item embs

next_item_eval = NextItemEvaluation.from_splits(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    exclude_unknowns=True,
    verbose=True,
    fmt="USIT",
    item_feature=FeatureModality(features=content, ids=item_ids),
)

models = [
    LETTER(  # lightweight example budget; use LETTER_BEAUTY_CONFIG to reproduce
        cf_embeddings=cf_embeddings,
        cf_embedding_ids=item_ids,
        cf_weight=0.02,
        diversity_weight=1e-3,
        rqvae_num_levels=4,
        rqvae_codebook_size=256,
        rqvae_latent_dim=32,
        rqvae_n_epochs=200,
        n_epochs=50,
        batch_size=256,
        max_len=20,
        scoring="beam",
        n_beams=50,
        device=DEVICE,
        verbose=True,
        seed=123,
    ),
    TIGER(  # baseline: same pipeline, plain RQ-VAE tokenizer
        rqvae_num_levels=4,
        rqvae_codebook_size=256,
        rqvae_latent_dim=32,
        rqvae_n_epochs=200,
        n_epochs=50,
        batch_size=256,
        max_len=20,
        scoring="beam",
        n_beams=50,
        device=DEVICE,
        verbose=True,
        seed=123,
    ),
]

metrics = [NDCG(k=10), NDCG(k=50), Recall(k=10), Recall(k=50), MRR()]

cornac.Experiment(
    eval_method=next_item_eval,
    models=models,
    metrics=metrics,
).run()
