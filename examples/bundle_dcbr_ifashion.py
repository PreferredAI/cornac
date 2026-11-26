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
"""DCBR on iFashion fixed splits, or ``--smoke`` for download-free CPU data.

Run from an installed checkout, for example:
``python examples/bundle_dcbr_ifashion.py --smoke``.
The full iFashion run benefits from a GPU and uses the authors' configuration.
"""

import argparse

import cornac
from cornac.datasets import bundle
from cornac.eval_methods import BundleEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import BundlePop, DCBR


def smoke_evaluation():
    """Build a tiny catalog with disjoint train, validation and test feedback."""
    return BundleEvaluation.from_splits(
        train_data=[
            (f"u{u}", f"b{(u + j) % 6}", 1.0) for u in range(4) for j in (0, 1)
        ],
        val_data=[(f"u{u}", f"b{(u + 2) % 6}", 1.0) for u in range(4)],
        test_data=[(f"u{u}", f"b{(u + 3) % 6}", 1.0) for u in range(4)],
        bundle_item_data=[
            (f"b{b}", f"i{(b + j) % 8}") for b in range(6) for j in (0, 1)
        ],
        test_mask="train",
        seed=2026,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Use tiny offline CPU data"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override training epochs"
    )
    args = parser.parse_args()
    if args.smoke:
        evaluation = smoke_evaluation()
        settings = dict(
            emb_size=8,
            num_layers=1,
            ub_layer_weights=(0.5, 0.5),
            ui_layer_weights=(0.5, 0.5),
            bi_layer_weights=(0.5, 0.5),
            diff_steps=3,
            diff_dims=(8,),
            diff_batch_size=4,
            batch_size=4,
            num_epochs=2,
        )
        k = 3
    else:
        evaluation = BundleEvaluation.from_splits(
            train_data=bundle.load_feedback("iFashion", split="train"),
            val_data=bundle.load_feedback("iFashion", split="validation"),
            test_data=bundle.load_feedback("iFashion", split="test"),
            bundle_item_data=bundle.load_bundle_item("iFashion"),
            user_item_data=bundle.load_user_item("iFashion"),
            test_mask="train",
            verbose=True,
            **bundle.load_ids("iFashion"),
        )
        settings = dict(use_gpu=args.device == "cuda")
        k = 20
    if args.epochs is not None:
        settings["num_epochs"] = args.epochs
    dcbr = DCBR(seed=2026, verbose=True, **settings)
    cornac.Experiment(
        eval_method=evaluation,
        models=[BundlePop(popularity="user"), dcbr],
        metrics=[Recall(k=k), NDCG(k=k)],
    ).run()


if __name__ == "__main__":
    main()
