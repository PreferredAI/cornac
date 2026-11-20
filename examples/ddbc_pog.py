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

Uses the data released with the paper (``datasets.zip`` linked from
https://github.com/241416/DDBC). Set ``POG_DIR`` to its ``pog_dense`` folder,
which holds the bundles (``train.txt``, ``valid.txt``, ``test.txt``, one
``bundle, item, item, ...`` line per bundle), the CLHE item features
(``clhe.pt``) and the authors' codebook initialization (``clhe_weight.npy``).

As in the paper (Section 4.1, Appendix C.2), held-out bundles keep a window of
k = 4 consecutive items starting at a random position, and the first 2 are
given. The evaluation ranks and constructs over the whole item catalog, so
the numbers are not comparable with the candidate pool used in the paper.
"""

import os
import random

import numpy as np
import torch

import cornac
from cornac.data import FeatureModality
from cornac.eval_methods import BundleConstructionEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import DDBC, BundlePop
from cornac.models.ddbc import DDBC_POG_CONFIG

POG_DIR = os.environ.get("POG_DIR", "datasets/pog_dense")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")


def read_bundles(split, k=None, rng=None):
    """(bundle, item, position) rows. With ``k``, bundles shorter than k are
    dropped and the others keep k consecutive items starting at a position
    drawn uniformly by ``rng``."""
    data = []
    with open(os.path.join(POG_DIR, f"{split}.txt")) as f:
        for line in f:
            fields = line.strip().split(", ")
            bundle, items = f"{split}_{fields[0]}", fields[1:]
            if k is not None:
                if len(items) < k:
                    continue
                start = rng.randint(0, len(items) - k)
                items = items[start : start + k]
            data.extend((bundle, item, pos) for pos, item in enumerate(items))
    return data


features = (
    torch.load(os.path.join(POG_DIR, "clhe.pt"), map_location="cpu").float().numpy()
)
codebook_init = np.load(os.path.join(POG_DIR, "clhe_weight.npy"), allow_pickle=True)
item_ids = [str(i) for i in range(len(features))]
window_rng = random.Random(1)

bundle_eval = BundleConstructionEvaluation.from_splits(
    train_data=read_bundles("train"),
    val_data=read_bundles("valid", k=4, rng=window_rng),
    test_data=read_bundles("test", k=4, rng=window_rng),
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
    BundlePop(name="BundlePop-cooccurrence", popularity="cooccurrence"),
    DDBC(
        **DDBC_POG_CONFIG,
        codebook_init=codebook_init,
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
