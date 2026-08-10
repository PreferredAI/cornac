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
"""DiffGRM (masked-diffusion Semantic IDs) on Amazon Sports (2014).

This example follows the DiffGRM Sports data path: per-user leave-last-out
splitting, item text containing title/price/brand/categories/description,
Sentence-T5 content embeddings, PSE tokenization, OCN training, and CPD
decoding. The model overrides the paper-style configuration's per-view loss
and paper decoder with the released pooled-token loss and released decoder
used by the release-fidelity study.

This is an end-to-end runnable reference, not an artifact-identical
reproduction. PCA and FAISS outputs depend on library versions; the controlled
study uses frozen ``item_sids``. Cornac's standard experiment output is also
item-expanded, whereas the paper and released evaluator report SID-level
metrics. See ``cornac/models/diffgrm/README.md`` for the controlled results and
limitations.

Requires ``sentence-transformers`` in addition to the packages in
``cornac/models/diffgrm/requirements.txt``. This is a paper-scale experiment;
training and beam evaluation are intended for a GPU.
"""

import torch
from sentence_transformers import SentenceTransformer

import cornac
from cornac.data import FeatureModality
from cornac.datasets import amazon_review
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import MRR, NDCG, Recall
from cornac.models import DiffGRM
from cornac.models.diffgrm import DIFFGRM_SPORTS_CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

data = amazon_review.load_feedback(category="sports")
texts, item_ids = amazon_review.load_text(
    category="sports",
    include_description=True,
)

encoder = SentenceTransformer("sentence-t5-base", device=DEVICE)
features = encoder.encode(texts, batch_size=256, show_progress_bar=True)
del encoder  # release encoder memory before fitting DiffGRM
if DEVICE == "cuda":
    torch.cuda.empty_cache()

next_item_eval = NextItemEvaluation.leave_last_out(
    data=data,
    exclude_unknowns=True,
    verbose=True,
    item_feature=FeatureModality(features=features, ids=item_ids),
)

models = [
    DiffGRM(
        **{
            **DIFFGRM_SPORTS_CONFIG,
            "view_loss_reduction": "token_mean",
            "scoring": "released",
            "device": DEVICE,
            "verbose": True,
            "seed": 2024,
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
