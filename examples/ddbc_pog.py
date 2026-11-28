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
"""DDBC (discrete diffusion for bundle construction) on POG outfits, k = 4.

Uses the data released with the paper (``datasets.zip``, linked from
https://github.com/LiAi16/DDBC). Set ``DDBC_DATA_DIR`` to the unzipped folder.
Its ``pog_dense`` folder holds the bundles (``train.txt``, ``valid.txt``,
``test.txt``, one ``bundle, item, item, ...`` line per bundle), the CLHE item
features (``clhe.pt``), and the released item IDs (``clhe_token.json``) with
their codebooks (``clhe_weight.npy``). Set ``DDBC_CKPT`` to the released
``pog_dense.ckpt`` (https://huggingface.co/liaialley/DDBC) to evaluate that
checkpoint instead of training the release recipe.

Held-out bundles are cut to their first 4 items and the first 2 are given,
as in the released test files.
"""

import json
import os

import numpy as np
import torch

import cornac
from cornac.data import FeatureModality
from cornac.eval_methods import BundleConstructionEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import DDBC, BundlePop
from cornac.models.ddbc import DDBC_PAPER_POG_CONFIG, DDBC_RELEASE_POG_CONFIG

POG_DIR = os.path.join(os.environ.get("DDBC_DATA_DIR", "datasets"), "pog_dense")
CKPT = os.environ.get("DDBC_CKPT")
K = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")


def read_bundles(split, k=None):
    """(bundle, item, position) rows. With ``k``, bundles shorter than k are
    dropped and the others are cut to their first k items."""
    data = []
    with open(os.path.join(POG_DIR, f"{split}.txt")) as f:
        for line in f:
            fields = line.strip().split(", ")
            bundle, items = f"{split}_{fields[0]}", fields[1:]
            if k is not None:
                if len(items) < k:
                    continue
                items = items[:k]
            data.extend((bundle, item, pos) for pos, item in enumerate(items))
    return data


features = (
    torch.load(os.path.join(POG_DIR, "clhe.pt"), map_location="cpu", weights_only=True)
    .float()
    .numpy()
)
item_ids = [str(i) for i in range(len(features))]
with open(os.path.join(POG_DIR, "clhe_token.json")) as f:
    item_sids = json.load(f)
codebooks = np.load(os.path.join(POG_DIR, "clhe_weight.npy"))

bundle_eval = BundleConstructionEvaluation.from_splits(
    train_data=read_bundles("train"),
    val_data=read_bundles("valid", k=K),
    test_data=read_bundles("test", k=K),
    fmt="BIT",
    input_ratio=0.5,
    input_order="first",
    item_ids=item_ids,
    exclude_unknowns=False,
    seed=1,
    verbose=True,
    item_feature=FeatureModality(features=features, ids=item_ids),
)

models = [
    BundlePop(name="BundlePop-item", popularity="item"),
    BundlePop(name="BundlePop-cooccurrence", popularity="cooccurrence"),
    DDBC(
        name="DDBC-release",
        **DDBC_RELEASE_POG_CONFIG,
        item_sids=item_sids,
        codebooks=codebooks,
        pretrained=CKPT,
        device=DEVICE,
        verbose=True,
        seed=1,
    ),
    DDBC(
        name="DDBC-paper",
        **DDBC_PAPER_POG_CONFIG,
        device=DEVICE,
        verbose=True,
        seed=1,
    ),
]

cornac.Experiment(
    eval_method=bundle_eval,
    models=models,
    metrics=[Recall(k=20), NDCG(k=20)],
).run()
