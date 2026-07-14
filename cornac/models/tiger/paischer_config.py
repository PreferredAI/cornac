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
"""Paischer et al. recipe for TIGER (arXiv 2412.08604) - the best documented
reproduction of the original paper's numbers (matches Beauty/Sports).

Taken from ``configs/setting/MenderTok_Beauty.yaml`` in
https://github.com/facebookresearch/preference_discerning (their TIGER
baseline shares the trainer). Slower than :data:`GRID_CONFIG` but tuned for
accuracy. Their item embeddings are Sentence-T5-XXL on title/price/brand/
categories text (descriptions instead for Toys) - the embedding side lives in
the experiment script, not here.

Usage::

    from cornac.models.tiger import PAISCHER_CONFIG, TIGER

    model = TIGER(**{**PAISCHER_CONFIG, "seed": 123})

Verbatim from their config: RQ-VAE (hidden 768/512/256, latent 128, AdamW wd
0.1, lr 1e-3, batch 2048, 8000 epochs, standardized inputs), seq2seq (6+6
layers, dropout 0.2, lr 3e-4 cosine with 10k warmup steps, weight decay
0.035, batch 64), beam width 30. Adapted to cornac's epoch-based trainer:
they cap training at 200k optimizer *steps* (one step = one minibatch update,
so ``n_steps = n_epochs * ceil(n_train_samples / batch_size)``, where a
session of length T contributes T-1 samples) with step-based early stopping
(patience 15). At batch 64 on Amazon-Beauty-scale data (~128k training
samples, ~2k steps/epoch), ``n_epochs=100`` lands near their 200k-step cap,
with best-on-val checkpoint selection instead of early stopping
(``val_eval_every``/``val_sample`` are pragmatic choices, not theirs).
"""

PAISCHER_CONFIG = dict(
    tokenizer="rqvae",
    feature_standardize=True,
    rqvae_hidden_dims=(768, 512, 256),
    rqvae_latent_dim=128,
    rqvae_n_epochs=8000,
    rqvae_batch_size=2048,
    rqvae_weight_decay=0.1,
    num_enc_layers=6,
    num_dec_layers=6,
    dropout=0.2,
    learning_rate=3e-4,
    lr_schedule="cosine",
    warmup_steps=10000,
    weight_decay=0.035,
    batch_size=64,
    n_beams=30,
    model_selection="best",
    val_metric="ndcg",
    val_k=10,
    n_epochs=100,
    val_eval_every=5,
    val_sample=2000,
)
