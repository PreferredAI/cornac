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
"""Released LETTER tokenizer and LETTER-TIGER training configurations."""

# Shared settings from RQ-VAE/main.py, LETTER-TIGER/ckpt/TIGER/config.json,
# LETTER-TIGER/utils.py, and LETTER-TIGER/run_train.sh. On one GPU, four
# accumulated 256-example minibatches reproduce the published two-GPU
# effective batch (2 devices x 256 x 2 accumulation = 1024 examples).
LETTER_CONFIG = {
    "feature_standardize": False,
    "rqvae_num_levels": 4,
    "rqvae_codebook_size": 256,
    "rqvae_latent_dim": 32,
    "rqvae_hidden_dims": (2048, 1024, 512, 256, 128, 64),
    "rqvae_beta": 0.25,
    "rqvae_quant_loss_weight": 1.0,
    "rqvae_sk_epsilon": 0.003,
    "rqvae_sk_iters": 50,
    "rqvae_kmeans_jobs": 10,
    "rqvae_learning_rate": 1e-3,
    "rqvae_batch_size": 1024,
    "rqvae_weight_decay": 1e-4,
    "rqvae_n_epochs": 10000,
    "n_clusters": 10,
    "collision_resolve_iters": 20,
    # Paper-wide recommended regularization values. The released Beauty
    # command overrides these below.
    "cf_weight": 0.02,
    "diversity_weight": 1e-3,
    # Released 4+4 T5 and generation recipe.
    "d_model": 128,
    "d_ff": 1024,
    "d_kv": 64,
    "num_heads": 6,
    "num_enc_layers": 4,
    "num_dec_layers": 4,
    "dropout": 0.1,
    "letter_base_vocab_size": 32100,
    "ranking_temperature": 1.0,
    "max_len": 20,
    "n_epochs": 200,
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "batch_size": 256,
    "gradient_accumulation_steps": 4,
    "lr_schedule": "cosine",
    "warmup_ratio": 0.01,
    "model_selection": "best",
    "val_eval_every": 1,
    "val_sample": None,
    "val_batch_size": 256,
    "early_stopping_patience": 20,
    "max_grad_norm": 1.0,
    "scoring": "beam",
    "n_beams": 20,
    "scoring_batch_size": 256,
}


# RQ-VAE/tokenize.sh selects the Beauty tokenizer trained for 10,000 epochs
# with alpha=0.1 and beta=1e-4.
LETTER_BEAUTY_CONFIG = {
    **LETTER_CONFIG,
    "cf_weight": 0.1,
    "diversity_weight": 1e-4,
}
