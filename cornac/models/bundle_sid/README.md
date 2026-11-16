# BundleRPG: CLHE-style CF bundle features with RPG

This is a **CF-only adaptation**, not a replication of multimodal CLHE. It ranks
existing catalog bundles and keeps `cornac/models/rpg/*` unchanged.

1. Learn constituent-item embeddings using LightGCN and BPR on supplied user-item
   feedback, or training user-bundle feedback times catalog membership if UI is
   absent. Supplied UI must have appropriate provenance; the loader cannot prove
   that external feedback was collected independently of held-out bundle labels.
2. Pool member vectors using `encoder="mean"`, or train a CLHE-style item/bundle
   contrastive encoder using feature masking, member dropping, self-attention,
   and position-free pooling (`encoder="clhe"`, the default).
3. Make one or several seeded permutations of each user's **training** bundles.
   These orders are synthetic. RPG learns next-bundle SID prediction on them.
4. Use RPG's existing OPQ and generator and exact scoring over every bundle.

There are no text/image/audio features in this implementation. The `random`
encoder is a random-bundle-vector control. `item_features` can supply an aligned,
training-safe feature matrix; the caller owns its provenance.

## Usage

Install Cornac normally, then install the optional requirements in this directory.
FAISS is required for real OPQ training; the current RPG algorithm uses exactly
256 entries per codebook. At least 256 training-observed bundles are needed and
FAISS recommends substantially more. Match the effective feature width to an
integer multiple of `n_codebook` (default width 64, digits 8, PCA disabled).

```python
from cornac.models import BundleRPG

model = BundleRPG(
    encoder="clhe", seed=42,
    cf_params={"num_epochs": 10},
    encoder_params={"num_epochs": 20, "max_bundle_items": 128},
    rpg_params={"n_codebook": 8, "d_model": 128, "n_head": 4},
    epoch_budgets=(10, 20),
)
model.fit(bundle_eval.train_set, bundle_eval.val_set)
scores = model.score(user_idx=0)  # aligned to bundle_eval.train_set.bid_map
```

The outer evaluation must be `BundleEvaluation` with author fixed splits, metrics
Recall/NDCG@20, and explicit `test_mask="train"` for the CrossCBR protocol.

```powershell
python examples/bundle_rpg_youshu.py --smoke --encoders mean clhe
python examples/bundle_rpg_youshu.py --encoders random mean clhe --epochs 10 20
```

The smoke run uses a training-selected user subset with the original fixed
split labels and the full catalog. Its results check integration; they are not
full-dataset benchmark results. Use multiple seeds and tuned CF baselines for
research claims. `--cache-dir` selects the normal Cornac dataset cache, and
`--output` selects the JSON report.

## Validation and reproducibility

`epoch_budgets` trains independent, seeded RPG fits and selects the best budget
on **masked full-catalog validation NDCG**. RPG's built-in last-target session
validation is disabled. Features are shared across budgets. With no validation,
only the largest budget is fitted. The selected model's test labels are not
used by `fit`. Existing RPG training, losses, and decoding are unchanged.

Constituent and bundle ID maps are distinct. The adapter pre-registers the
complete bundle catalog, including catalog-only bundles, and retains a fixed
training-only inference history. Users with no history receive training bundle
popularity scores; singleton users do not produce next-item training examples.

`training_diagnostics_` records ID/training-edge hashes, feature source,
empty/singleton/truncated histories, SID collisions, and distinct codes per
digit. `encoder_diagnostics_` stores training losses, the number of capped
bundles, empty bundles, and encoder weights. Large bundles are deterministically
subsampled to `max_bundle_items` for CLHE-style training and export; mean pooling
uses all members. This difference must be controlled when studying large bundles.
Model save/load retains predictions; `clone` retains initialization settings.

All catalog membership is treated as transductive metadata. Optional RPG PCA and
standardization fit on all catalog features; OPQ uses RPG's training-observed
bundle mask. This is not an inductive cold-bundle protocol. Exact scoring still
builds RPG's existing adjacency graph during fitting, so its cost remains.

Tests exercise a real RPG backbone using deterministic fixture codes to keep
small tests fast. A real-FAISS pilot is needed to verify the entire tokenizer
path. Optional dependencies load during fitting, not when importing Cornac.

Research design: `research/docs/ideas/2026-11-10-clhe-opq-rpg-bundle-sid.md`.
