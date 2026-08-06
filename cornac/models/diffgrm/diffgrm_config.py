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
"""Paper-style DiffGRM configurations for Amazon-2014 experiments."""

DIFFGRM_CONFIG = dict(
    n_digit=4,
    codebook_size=256,
    pca_dim=256,
    max_len=50,
    min_history=2,
    encoder_n_layer=1,
    decoder_n_layer=4,
    n_inner=1024,
    dropout=0.1,
    masking_strategy="guided",
    confidence_method="msp",
    n_views=4,
    view_loss_reduction="view_mean",
    scoring="paper",
    n_epochs=100,
    batch_size=1024,
    weight_decay=0.0,
    lr_schedule="cosine",
    warmup_steps=10000,
    max_grad_norm=1.0,
    model_selection="best",
    val_k=10,
    val_batch_size=32,
    val_beam_size=32,
    val_eval_every=1,
    early_stopping_patience=15,
    val_sample=None,
)

DIFFGRM_SPORTS_CONFIG = dict(
    DIFFGRM_CONFIG,
    d_model=256,
    n_head=4,
    learning_rate=0.003,
    label_smoothing=0.1,
    beam_size=128,
    val_eval_start=20,
)

DIFFGRM_BEAUTY_CONFIG = dict(
    DIFFGRM_CONFIG,
    d_model=256,
    n_head=4,
    learning_rate=0.01,
    label_smoothing=0.1,
    beam_size=256,
    val_eval_start=20,
)

DIFFGRM_TOYS_CONFIG = dict(
    DIFFGRM_CONFIG,
    d_model=1024,
    n_head=8,
    learning_rate=0.003,
    label_smoothing=0.15,
    beam_size=128,
    val_eval_start=10,
)

# Tiny architecture for unit/scheduler smoke tests. It requires precomputed
# ``item_sids`` because 4-way codes are not the paper's 8-bit PSE tokenizer.
DIFFGRM_SMOKE_CONFIG = dict(
    n_digit=2,
    codebook_size=4,
    d_model=32,
    encoder_n_layer=1,
    decoder_n_layer=1,
    n_head=4,
    n_inner=64,
    dropout=0.0,
    max_len=5,
    min_history=1,
    n_views=2,
    n_epochs=1,
    batch_size=4,
    learning_rate=1e-3,
    lr_schedule="constant",
    warmup_steps=0,
    scoring="catalog",
    beam_size=8,
    model_selection="last",
)
