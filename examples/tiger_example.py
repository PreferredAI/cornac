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
"""TIGER (generative retrieval with semantic IDs) on Amazon Beauty (2014).

Reproduces the TIGER paper setup: 5-core Amazon Beauty reviews with per-user
leave-last-out splitting (the paper's protocol; see
``NextItemEvaluation.from_timestamps`` for a leakage-free alternative), item
content text (title/price/brand/categories) embedded with Sentence-T5, and
the Paischer et al. training recipe shipped as
``cornac.models.tiger.PAISCHER_CONFIG`` -- the best documented reproduction
of the paper's numbers.

Requires ``sentence-transformers`` on top of the model requirements
(torch, transformers).

Expected results (test split, beam scoring): Recall@5 ~= 0.042,
NDCG@5 ~= 0.027, vs 0.0454 / 0.0321 reported in the paper -- on par with the
best published reproductions. Training takes about an hour on one GPU;
evaluation decodes a beam per test user and takes a comparable amount of time.
"""

import torch
from sentence_transformers import SentenceTransformer

import cornac
from cornac.data import FeatureModality
from cornac.datasets import amazon_review
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import TIGER
from cornac.models.tiger import PAISCHER_CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

data = amazon_review.load_feedback(category="beauty")
texts, item_ids = amazon_review.load_text(category="beauty")

encoder = SentenceTransformer("sentence-t5-base", device=DEVICE)
features = encoder.encode(texts, batch_size=64, show_progress_bar=True)
del encoder  # release encoder memory before training
if DEVICE == "cuda":
    torch.cuda.empty_cache()

next_item_eval = NextItemEvaluation.leave_last_out(
    data=data,
    exclude_unknowns=True,
    verbose=True,
    item_feature=FeatureModality(features=features, ids=item_ids),
)

models = [
    TIGER(
        **{
            **PAISCHER_CONFIG,
            "device": DEVICE,
            "verbose": True,
            "seed": 123,
        }
    ),
]

metrics = [
    Recall(k=5),
    Recall(k=10),
    NDCG(k=5),
    NDCG(k=10),
    MRR(),
]

cornac.Experiment(
    eval_method=next_item_eval,
    models=models,
    metrics=metrics,
).run()
