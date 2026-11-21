# DDBC

Cornac implementation of **DDBC** (Discrete Diffusion for Bundle Construction, Tu et al., ICLR 2026, [OpenReview](https://openreview.net/forum?id=dKyhgfe50H), [code](https://github.com/241416/DDBC)): a residual vector quantizer turns item features into semantic IDs (3 levels of codes plus a suffix for colliding items), and a diffusion transformer (DiT) learns to unmask the semantic-ID tokens of bundles. A partial bundle is completed by unmasking the IDs of its missing items given those of its seed items.

## Requirements

```bash
pip install -r cornac/models/ddbc/requirements.txt
```

## Usage

DDBC is a `BundleConstructionRecommender` evaluated with `BundleConstructionEvaluation`. It needs item features, supplied as a `FeatureModality`, and bundles in the `BIT` format (bundle, item, position of the item in the bundle). See [`examples/ddbc_pog.py`](../../../examples/ddbc_pog.py) for an end-to-end run:

```python
from cornac.data import FeatureModality
from cornac.eval_methods import BundleConstructionEvaluation
from cornac.experiment import Experiment
from cornac.metrics import NDCG, Recall
from cornac.models import DDBC
from cornac.models.ddbc import DDBC_POG_CONFIG

eval_method = BundleConstructionEvaluation.from_splits(
    train_data=train, val_data=val, test_data=test, fmt="BIT",
    input_ratio=0.5, input_order="first", item_ids=item_ids,
    exclude_unknowns=False, seed=1,
    item_feature=FeatureModality(features=item_features, ids=item_ids),
)
model = DDBC(**{**DDBC_POG_CONFIG, "device": "cuda", "seed": 1})
Experiment(eval_method, [model], [Recall(k=20), NDCG(k=20)]).run()
```

With `input_ratio > 0`, DDBC completes the given items of each bundle. With `input_ratio=0`, it constructs bundles from scratch by unmasking every item (Appendix D.1 of the paper). DDBC does not use the user of a query, so with `confidence` decoding every from-scratch query gets the same bundle.

The paper settings ship as configs. All of them use a 6-block DiT (hidden size 64, 8 heads), T = 25, confidence decoding (Algorithm 2), AdamW (lr 3e-4, 500 warm-up steps), batch 300 with 2 accumulation steps, 20k optimizer steps, EMA decay 0.9999 and the best checkpoint by validation NLL:

| Dataset | Config | `bundle_size` | `codebook_size` | `swap_ratio` |
| --- | --- | ---: | ---: | ---: |
| POG | `DDBC_POG_CONFIG` | 4 | 128 | 0.8 |
| Spotify | `DDBC_SPOTIFY30_CONFIG` / `DDBC_SPOTIFY60_CONFIG` / `DDBC_SPOTIFY90_CONFIG` | 30 / 60 / 90 | 256 | 0.4 |

## Implementation notes

- **`construct`** generates the semantic IDs of the missing items and returns one catalog item for each. This is the item with the generated ID, or the item closest by cosine to the sum of the generated codewords when no item has that ID. When the item is already in the bundle, the closest one that is not in it is taken instead. `Set-F1` and `Set-Jaccard` use these items.
- **`score`** serves the ranking metrics. The given items fill the first blocks of a window of `bundle_size` items and the others are masked. An item scores the sum of the log-probabilities of its semantic-ID tokens in the first masked block, with the diffusion step matching the masked fraction of the window.
- **Long bundles.** The model trains on windows of `bundle_size` items. For a longer held-out bundle, `construct` generates the whole sequence at once and `score` keeps the last `bundle_size - 1` given items.

## Main differences vs the reference code

- **Codebook initialization.** The reference code starts the quantizer from a released `clhe_weight.npy`. The default here is residual k-means on the item features. Pass `codebook_init=np.load("clhe_weight.npy")` to use the authors' initialization.
- **Evaluation.** The reference code maps generated items to a small candidate pool and reports F1, Jaccard and OAS. `BundleConstructionEvaluation` ranks and constructs over the whole catalog without negative sampling, so its numbers are not comparable with the paper.
- **Training objective.** Eq. 7 of the paper writes the masked-token cross-entropy without a weight, while its text calls it a weighted negative ELBO. The loss here weights each masked token by `T / t`, the negative ELBO under the linear schedule `alpha_t = 1 - t / T`. The reference code applies the same weight in continuous time.
- **Time steps.** Following Section 3.3, `t` is discrete in {1, ..., T} and the denoiser is conditioned on it. The reference code samples a continuous `t` and does not condition on it.
- **Decoding.** The default `decode="confidence"` is the confidence-ordered argmax unmasking of Algorithm 2, as described in the paper. The reference code samples ancestrally, which produced Table 1. Use `decode="ancestral"` for that. On POG, Algorithm 2 scores much higher.
- **Sampling.** Ancestral sampling uses a generator seeded from `seed`, the query and its given items, so a query gets the same bundle whatever the evaluation order.
- **Held-out windows.** The example keeps a random window of k items of each held-out bundle (Section 4.1). The reference code keeps the first k.
- **No wandb logging.**
- **Reproducibility.** Runs with the same `seed` are bitwise identical only with deterministic kernels, e.g. `torch.set_num_threads(1)` on CPU. Otherwise, multi-threaded float reductions leave differences around 1e-7, which ancestral sampling can amplify.

## Results

Test results of [`examples/ddbc_pog.py`](../../../examples/ddbc_pog.py) on POG:

| Model | NDCG@20 | Recall@20 | Set-F1 | Set-Jaccard | Train(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BundlePop-cooccurrence | **0.2684** | **0.3346** | **0.1843** | **0.1288** | 0.0118 |
| DDBC | 0.2160 | 0.2521 | 0.1323 | 0.0904 | 1373 |

**Setting:** the `pog_dense` data released with the paper (20,780 training bundles, 2,521 test queries, 31,217 items). Each held-out bundle keeps a random window of k = 4 items and the first 2 are given. DDBC uses `DDBC_POG_CONFIG` with the authors' `clhe_weight.npy` codebooks, `decode="confidence"` and `seed=1`, trained on a Kaggle GPU.

**A few things to note:**

- Ranking and construction cover the whole item catalog, so the numbers are not comparable with the candidate-pool results of the paper.
- In this setting, the co-occurrence baseline beats DDBC on every metric.
