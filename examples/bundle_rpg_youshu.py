# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""CF-only CLHE-style/mean/random bundle features with unchanged RPG.

Run from the repository root after installing Cornac and bundle_sid requirements:
    python examples/bundle_rpg_youshu.py --encoders mean clhe
    python examples/bundle_rpg_youshu.py --smoke --encoders clhe

The smoke run keeps fixed-split labels for a small user subset; its metrics are
integration diagnostics, not full-dataset benchmark results.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from cornac.datasets import bundle
from cornac.eval_methods import BundleEvaluation, ranking_eval
from cornac.metrics import NDCG, Recall
from cornac.models import BundlePop, BundleRPG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["Youshu", "NetEase", "iFashion"], default="Youshu")
    parser.add_argument("--encoders", nargs="+", choices=["mean", "clhe", "random"], default=["mean", "clhe"])
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--cf-epochs", type=int, default=10)
    parser.add_argument("--encoder-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-perm", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", default="output/bundle_rpg/results.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import torch
    import faiss

    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    ids = bundle.load_ids(args.dataset, cache_dir=args.cache_dir)
    splits = [bundle.load_feedback(args.dataset, split, cache_dir=args.cache_dir)
              for split in ("train", "validation", "test")]
    ui = bundle.load_user_item(args.dataset, cache_dir=args.cache_dir)
    if args.smoke:
        # Select using training feedback only, independently of held-out labels.
        counts = {}
        for user, _, _ in splits[0]:
            counts[user] = counts.get(user, 0) + 1
        eligible = [u for u in ids["user_ids"] if counts.get(u, 0) >= 2]
        rng = np.random.RandomState(args.seed)
        selected = set(rng.choice(eligible, min(256, len(eligible)), replace=False))
        ids["user_ids"] = [u for u in ids["user_ids"] if u in selected]
        splits = [[row for row in split if row[0] in selected] for split in splits]
        ui = [row for row in ui if row[0] in selected]
    evaluation = BundleEvaluation.from_splits(
        train_data=splits[0], val_data=splits[1], test_data=splits[2],
        bundle_item_data=bundle.load_bundle_item(args.dataset, cache_dir=args.cache_dir),
        user_item_data=ui, test_mask="train", seed=args.seed, **ids,
    )
    common = dict(
        emb_size=32 if args.smoke else 64, seed=args.seed, device=args.device,
        epoch_budgets=(1,) if args.smoke else tuple(args.epochs), n_perm=args.n_perm,
        cf_params=dict(num_epochs=1 if args.smoke else args.cf_epochs),
        encoder_params=dict(num_epochs=1 if args.smoke else args.encoder_epochs,
                            max_bundle_items=64 if args.smoke else 128),
        rpg_params=dict(n_codebook=4 if args.smoke else 8, d_model=32 if args.smoke else 128,
                        n_layer=1 if args.smoke else 2, n_head=4, n_inner=128 if args.smoke else 256,
                        max_len=20 if args.smoke else 50, batch_size=64,
                        graph_edges=10, embd_dropout=0.1, attn_dropout=0.1),
        verbose=True,
    )
    report = dict(dataset=args.dataset, smoke=args.smoke, seed=args.seed,
                  protocol="fixed author splits; full catalog; test_mask=train",
                  users=len(ids["user_ids"]), bundles=len(ids["bundle_ids"]),
                  split_rows=[len(split) for split in splits],
                  configuration=common, models={})
    metrics = [Recall(k=20), NDCG(k=20)]
    models = [BundlePop()] + [BundleRPG(encoder=encoder, name="BundleRPG_" + encoder, **common)
                             for encoder in args.encoders]
    for model in models:
        started = time.perf_counter()
        model.fit(evaluation.train_set, evaluation.val_set)
        train_seconds = time.perf_counter() - started
        started = time.perf_counter()
        values = ranking_eval(model, metrics, evaluation.train_set, evaluation.test_set)[0]
        info = dict(recall20=values[0], ndcg20=values[1],
                    train_seconds=train_seconds, scoring_seconds=time.perf_counter() - started)
        if isinstance(model, BundleRPG):
            info.update(best_epoch=model.best_epoch, validation=model.val_history,
                        diagnostics=model.training_diagnostics_,
                        encoder_losses=model.encoder_diagnostics_.get("losses", []),
                        truncated_bundles=model.encoder_diagnostics_.get("truncated_bundles", 0))
        report["models"][model.name] = info
        print(model.name, json.dumps(info))
        # Retain partial results if another model later fails.
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
