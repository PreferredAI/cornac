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

"""Paper configurations of DDBC on POG and Spotify.

Section 4.1 and Appendices B and C.1 of the paper give the architecture, T,
the number of steps, the bundle lengths and the swap ratios. The codebook sizes
are those of the released ``clhe_weight.npy`` files. The optimizer settings
follow the reference code, since the paper does not state them.
"""

DDBC_CONFIG = dict(
    n_codebooks=3,
    hidden_size=64,
    cond_dim=128,
    n_blocks=6,
    n_heads=8,
    dropout=0.1,
    n_diffusion_steps=25,
    decode="confidence",
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

DDBC_POG_CONFIG = dict(DDBC_CONFIG, bundle_size=4, codebook_size=128, swap_ratio=0.8)

DDBC_SPOTIFY30_CONFIG = dict(
    DDBC_CONFIG, bundle_size=30, codebook_size=256, swap_ratio=0.4
)

DDBC_SPOTIFY60_CONFIG = dict(DDBC_SPOTIFY30_CONFIG, bundle_size=60)

DDBC_SPOTIFY90_CONFIG = dict(DDBC_SPOTIFY30_CONFIG, bundle_size=90)
