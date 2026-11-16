# CLHE-style bundle features with unchanged RPG

Implemented on `BunRec`, based on the research idea at
[`research/docs/ideas/2026-11-10-clhe-opq-rpg-bundle-sid.md`](../research/docs/ideas/2026-11-10-clhe-opq-rpg-bundle-sid.md).
The CoGen four-channel idea is outside this implementation.

## Scope delivered

- Constituent-item LightGCN/BPR features from supplied UI or training UB @ BI.
- Mean and random bundle-feature controls.
- CF-only CLHE-style feature projection, member attention, item-level feature
  masking contrast, and bundle-level member-drop contrast.
- Complete-catalog ID alignment and seeded, training-only USIT sessions.
- `BundleRPG(BundleRecommender)` wrapping the existing RPG model, with exact
  catalog scores and a popularity fallback for empty histories.
- External epoch-budget selection on validation bundle NDCG, keeping RPG's
  generator, loss, tokenizer and decoder source unchanged.
- Focused tests, a configurable Youshu/NetEase/iFashion runner, diagnostics,
  optional dependency requirements, and model save/load coverage.

The implementation and usage are documented in
[`cornac/models/bundle_sid/README.md`](../cornac/models/bundle_sid/README.md).

## Validation performed

The new model tests cover seeded adapter behavior, full catalog/constituent ID
separation, train-only histories, exact-complement negative sampling, CF feature
determinism, encoder training/member-order invariance, masked bundle validation,
real RPG backbone training with deterministic fixture codes, save/load and clone.
The focused regression run passed 43 checks with two existing optional-runtime
checks skipped; an additional early-rejection check covers undersized OPQ data.
A separate real-FAISS pilot ran both mean and CLHE-style models on 256 users
selected from Youshu training feedback, retaining the original author split
labels and all 4,771 catalog bundles. Each training stage ran one epoch.

| Pilot model | Recall@20 | NDCG@20 |
|-------------|----------:|--------:|
| BundlePop | 0.001497 | 0.001066 |
| BundleRPG mean | 0.024360 | 0.014281 |
| BundleRPG CLHE-style | 0.008252 | 0.003931 |

These are one-seed subset integration diagnostics, not tuned benchmark results.
The CLHE-style model did not beat mean pooling in this pilot. FAISS warned that
1,281 train-observed bundle vectors are below its recommended 9,984 examples
for 256-way clustering. CLHE-style pooling capped 807 catalog bundles at 64
members in this smoke configuration; 51 users exceeded the 20-bundle history
limit. Default full settings differ and must be evaluated independently.

## Runtime note

The original source checkout lacked compiled Cornac extensions. Tests used the
local Python source with compiled extensions from a temporary Cornac 3.2.0
Windows wheel, plus temporary FAISS 1.16.0, under the ignored `tmp/` directory.
This verifies the changed Python modules against that runtime; it is not a
clean source-build verification of all Cornac native extensions. A normal
installed development environment should run the same tests with native
extensions built for this checkout.

## Follow-up experiment plan

1. Reproduce BundlePop and tuned CrossCBR on full Youshu with `test_mask="train"`.
2. Run random/mean/CLHE-style features with the same RPG budgets and three seeds.
3. Control member truncation, feature width, OPQ size, history length and
   permutation count; compare encoder quality against mean, not only popularity.
4. Expand promising comparisons to five seeds and NetEase. Measure full feature
   training, OPQ, adjacency construction, scoring time and storage.
5. Keep accuracy claims conditional on consistent gains over matched mean
   pooling and competitive tuned CF results. Content modalities require a new
   provenance/alignment audit and are not present in this CF-only adaptation.

Commands in the model README run the implemented pipeline. Full benchmark and
multi-seed experiments remain follow-up work, not a prerequisite for using the
implementation.
