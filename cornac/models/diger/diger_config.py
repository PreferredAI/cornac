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

"""Beauty validation-selected recipes for retained ST5/OpenAI item embeddings.

Both conditions selected the same learning rates from six candidates. The
retained study used a seed-2024 initializer and joint seeds 2025/2026/2027.
These dictionaries train a fresh initializer unless one is explicitly supplied;
they do not reproduce the retained study assets by themselves. See README.md.
"""

BEAUTY_ST5_CONFIG = {
    "n_epochs": 120,
    "batch_size": 256,
    "eval_batch_size": 32,
    "learning_rate": 5e-4,
    "tokenizer_learning_rate": 3e-5,
    "weight_decay": 0.05,
    "warmup_steps": 500,
    "max_len": 210,
    "n_beams": 20,
    "early_stopping_patience": 15,
    "initializer_seed": 2024,
    "seed": 2025,
}

# Architecture and selected rates coincide; feature widths differ (768/3072).
BEAUTY_OPENAI_CONFIG = dict(BEAUTY_ST5_CONFIG)
