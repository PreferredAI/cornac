# Copyright 2026 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Recipes of DDBC on POG and Spotify.

Usage::

    from cornac.models.ddbc import DDBC, DDBC_RELEASE_POG_CONFIG

    model = DDBC(**{**DDBC_RELEASE_POG_CONFIG, "seed": 1})

``DDBC_PAPER_*`` follows the paper and ``DDBC_RELEASE_*`` the released code
(LiAi16/DDBC at e1f4afc5). To compare with the released checkpoints, also
pass the released ``item_sids`` and ``codebooks``.
"""

_DDBC_CONFIG = dict(
    n_codebooks=3,
    hidden_size=64,
    cond_dim=128,
    n_blocks=6,
    n_heads=8,
    dropout=0.1,
    batch_size=300,
    accumulate=2,
    max_steps=20000,
    val_every=32,
    learning_rate=3e-4,
    weight_decay=0.0,
    warmup_steps=500,
    grad_clip=1.0,
    ema_decay=0.9999,
    model_selection="best",
)

DDBC_PAPER_CONFIG = dict(
    _DDBC_CONFIG,
    tokenizer="rvq",
    codebook_size=128,
    fallback_levels=3,
    layout="joint",
    n_diffusion_steps=25,
    time_conditioning=True,
    decode="confidence",
    sampling_steps=25,
)
DDBC_PAPER_POG_CONFIG = dict(DDBC_PAPER_CONFIG, bundle_size=4, swap_ratio=0.8)
DDBC_PAPER_SPOTIFY30_CONFIG = dict(DDBC_PAPER_CONFIG, bundle_size=30, swap_ratio=0.4)
DDBC_PAPER_SPOTIFY60_CONFIG = dict(DDBC_PAPER_SPOTIFY30_CONFIG, bundle_size=60)
DDBC_PAPER_SPOTIFY90_CONFIG = dict(DDBC_PAPER_SPOTIFY30_CONFIG, bundle_size=90)

DDBC_RELEASE_CONFIG = dict(
    _DDBC_CONFIG,
    tokenizer="faiss",
    fallback_levels=2,
    layout="split",
    n_diffusion_steps=None,
    time_conditioning=False,
    decode="ancestral",
    sampling_steps=256,
)
DDBC_RELEASE_POG_CONFIG = dict(
    DDBC_RELEASE_CONFIG, bundle_size=4, codebook_size=128, swap_ratio=0.8
)
DDBC_RELEASE_SPOTIFY30_CONFIG = dict(
    DDBC_RELEASE_CONFIG, bundle_size=30, codebook_size=256, swap_ratio=0.4
)
DDBC_RELEASE_SPOTIFY60_CONFIG = dict(DDBC_RELEASE_SPOTIFY30_CONFIG, bundle_size=60)
DDBC_RELEASE_SPOTIFY90_CONFIG = dict(DDBC_RELEASE_SPOTIFY30_CONFIG, bundle_size=90)
