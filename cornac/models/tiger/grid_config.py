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
"""GRID handbook recipe for TIGER (Ju et al., 2025, arXiv 2507.22224).

Trades some accuracy for much faster training: the residual k-means tokenizer
needs no gradient training at all (GRID Table 1 finds it *outperforms* RQ-VAE
anyway), and the seq2seq stage runs a short epoch budget with best-on-val
checkpoint selection instead of a long fixed schedule.

Usage::

    from cornac.models.tiger import GRID_CONFIG, TIGER

    model = TIGER(**{**GRID_CONFIG, "seed": 123})

Verbatim from the GRID paper: optimizer settings (Adam lr 5e-4, weight decay
1e-6, batch 256, constant lr), NDCG@10 validation, rkmeans tokenizer with the
paper's (3 levels, 256 codes) shape and the paper's 4+4-layer transformer
(both cornac defaults, so not repeated here). Adapted to cornac's epoch-based
trainer: GRID counts optimizer *steps* (one step = one minibatch update, i.e.
``n_steps = n_epochs * ceil(n_train_samples / batch_size)``, where a session
of length T contributes T-1 samples) and early-stops on step-based validation
intervals (every 100 steps, patience 10); cornac instead trains ``n_epochs``
and keeps the best-on-val checkpoint, so ``n_epochs``/``val_eval_every``/
``val_sample`` here are pragmatic equivalents, not GRID values.
"""

GRID_CONFIG = dict(
    tokenizer="rkmeans",
    learning_rate=5e-4,
    weight_decay=1e-6,
    batch_size=256,
    lr_schedule="constant",
    model_selection="best",
    val_metric="ndcg",
    val_k=10,
    n_epochs=50,
    val_eval_every=1,
    val_sample=2000,
)
