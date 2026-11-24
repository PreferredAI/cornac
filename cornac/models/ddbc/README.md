# DDBC

Cornac implementation of **DDBC** (Discrete Diffusion for Bundle Construction, Tu et al., ICLR 2026, [OpenReview](https://openreview.net/forum?id=dKyhgfe50H), [code](https://github.com/LiAi16/DDBC/tree/e1f4afc5), [checkpoints](https://huggingface.co/liaialley/DDBC/tree/f390ddead78114c703fae935c851c73eea04d8c1)): a residual quantizer turns item features into semantic IDs (3 levels of codes plus a suffix for colliding items), and a diffusion transformer (DiT) learns to unmask the semantic-ID tokens of bundles. A partial bundle is completed by unmasking the IDs of its missing items given those of its seed items. The DiT, the loss and the ancestral sampler follow [MDLM](https://github.com/kuleshov-group/mdlm) (Sahoo et al., NeurIPS 2024), which DDBC builds on.

## Requirements

```bash
pip install -r cornac/models/ddbc/requirements.txt
```

`faiss-cpu` is only needed for `tokenizer="faiss"`.

## Usage

DDBC is a `BundleConstructionRecommender` evaluated with `BundleConstructionEvaluation`. It needs item features, supplied as a `FeatureModality`, and bundles in the `BIT` format (bundle, item, position of the item in the bundle). See [`examples/ddbc_pog.py`](../../../examples/ddbc_pog.py) for an end-to-end run:

```python
from cornac.data import FeatureModality
from cornac.eval_methods import BundleConstructionEvaluation
from cornac.experiment import Experiment
from cornac.metrics import NDCG, Recall
from cornac.models import DDBC
from cornac.models.ddbc import DDBC_PAPER_POG_CONFIG, DDBC_RELEASE_POG_CONFIG

eval_method = BundleConstructionEvaluation.from_splits(
    train_data=train,
    val_data=val,
    test_data=test,
    fmt="BIT",
    input_ratio=0.5,
    input_order="first",
    item_ids=item_ids,
    exclude_unknowns=False,
    seed=1,
    item_feature=FeatureModality(features=item_features, ids=item_ids),
)
paper = DDBC(**{**DDBC_PAPER_POG_CONFIG, "device": "cuda", "seed": 1})
release = DDBC(
    **DDBC_RELEASE_POG_CONFIG,
    item_sids=released_item_sids,
    codebooks=released_codebooks,
    pretrained="pog_dense.ckpt",
    device="cuda",
    seed=1,
)
Experiment(eval_method, [paper, release], [Recall(k=20), NDCG(k=20)]).run()
```

With `input_ratio > 0`, DDBC completes the given items of each bundle. With `input_ratio=0`, it constructs bundles from scratch by unmasking every item. DDBC does not use the user of a query, so with `confidence` decoding every from-scratch query gets the same bundle.

### Recipes

One class covers both the paper and the released code, and both ship as configs. The constructor defaults follow the paper.

| Dataset | Paper config | Release config | `bundle_size` | `codebook_size` | `swap_ratio` |
| --- | --- | --- | ---: | ---: | ---: |
| POG | `DDBC_PAPER_POG_CONFIG` | `DDBC_RELEASE_POG_CONFIG` | 4 | 128 | 0.8 |
| Spotify | `DDBC_PAPER_SPOTIFY{30,60,90}_CONFIG` | `DDBC_RELEASE_SPOTIFY{30,60,90}_CONFIG` | 30 / 60 / 90 | 256 | 0.4 |

The paper does not state `codebook_size`, so both recipes use the release's values. The recipes differ in these arguments:

| Argument | Paper recipe | Release recipe |
| --- | --- | --- |
| `tokenizer` | `"rvq"`: residual quantizer with greedy encoding, codebooks trained by gradient descent | `"faiss"`: Faiss `IndexResidualQuantizer`, inner-product metric, beam search |
| `fallback_levels` | 3: an ID that names no item is matched through the sum of its codewords | 2: levels 1 and 2 only |
| `layout` | `"joint"`: given items and masked slots in one `<bos> ... <eos>` sequence | `"split"`: `<bos>` given `<eos>` `<bos>` slots `<eos>` |
| `n_diffusion_steps` | 25: discrete t in {1, ..., T}, mask rate t / T | `None`: continuous t in [0.001, 1] |
| `time_conditioning` | `True`: the denoiser sees t | `False`: it sees 0 |
| `decode` | `"confidence"`: most confident tokens first | `"ancestral"`: the release's sampler |
| `sampling_steps` | 25 | 256 |

Both recipes share the paper's DiT (6 blocks, hidden size 64, 8 heads) with the release's conditioning size 128 and dropout 0.1, the paper's 20,000 optimizer steps, and the released code's training settings, which the paper does not state: AdamW at 3e-4 with 500 warm-up steps and then a constant rate, clipping 1.0, EMA decay 0.9999, validation every 32 batches, and batch 300 with 2 accumulation steps. Both keep the averaged weights with the lowest validation loss.

The `"rvq"` tokenizer starts from residual k-means and trains its codebooks with full-batch Adam. The paper gives none of these settings. Faiss training is random, so `tokenizer="faiss"` does not give the released IDs. To use the released checkpoints, pass the released `clhe_token.json` as `item_sids` and `clhe_weight.npy` as `codebooks`, as the example does.

## Implementation notes

- **Loss.** Each masked code token costs its cross-entropy times `T / t` in discrete time and `1 / t` in continuous time, the MDLM weights. The paper prints the cross-entropy without a weight.
- **Decoding.** A code position may only take codes of its own level. `"confidence"` reveals at each step the masked positions whose best token is most probable. `"ancestral"` follows the release's sampler, which draws each masked position with Gumbel-max.
- **`construct`** turns each generated ID into a vector, the vector of the item with that ID or, for an ID that names no item, the sum of its first `fallback_levels` codewords, and takes the free catalog item closest by cosine. `Set-F1` and `Set-Jaccard` use these items.
- **`score`** ranks from the same decoded bundle: the constructed items first, then every other item by its highest cosine similarity to the generated slots. It decodes `bundle_size - len(given)` slots, so held-out bundles should have `bundle_size` items, as in the release.
- **Released checkpoints.** `pretrained` reads a released `.ckpt` without running code from it, loads its EMA weights and skips training. It requires `item_sids` and `codebooks`.
- **Sampling.** The decoding generator is seeded from `seed` and the query, so a query gets the same bundle whatever the evaluation order.

## Main differences vs the reference code

- **Evaluation.** `BundleConstructionEvaluation` ranks and constructs over the whole catalog, so its numbers are not comparable with those of the paper. The release scores each test bundle on a candidate pool of the targets plus 99 random items per target, which can include the given items.
- **Held-out bundles.** As in the released test files, held-out bundles are cut to their first k items, and the first ceil(k / 2) are given. With random windows of k items instead, BundlePop-cooccurrence reaches Set-F1 0.1843 on POG against 0.1656 here.
- **Training length and batch.** Both recipes train 20,000 steps on batches of 300 with 2 accumulation steps. The released POG, Spotify30 and Spotify60 checkpoints store `max_steps: 30000` and a global batch of 512, 256 and 512, and were saved at steps 10,375, 21,312 and 14,431. `max_steps`, `batch_size` and `accumulate` remain arguments.
- **Precision.** The released checkpoints were trained in bfloat16. The recipes keep `use_bf16=False` so that they run on any GPU.
- **Validation masks.** The validation loss draws its masks with a fixed seed, so checkpoints are compared on the same masks. The release draws new masks at every validation.
- **Reproducibility.** Two GPU trainings of the paper recipe with seed 1 gave POG Set-F1 0.1059 and 0.1091. Decoding a released checkpoint twice gave identical numbers.

## Results

Test results of [`examples/ddbc_pog.py`](../../../examples/ddbc_pog.py) on POG:

| Model | NDCG@20 | Recall@20 | Set-F1 | Set-Jaccard | Train(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BundlePop-item | 0.1445 | 0.2237 | 0.0841 | 0.0561 | 0.0148 |
| BundlePop-cooccurrence | **0.2472** | **0.3167** | **0.1656** | **0.1164** | 0.0047 |
| DDBC-release, released checkpoint, `sampling_steps=256` | 0.0683 | 0.0817 | 0.0706 | 0.0477 | 0.5386 |
| DDBC-release, released checkpoint, `sampling_steps=25` | 0.0646 | 0.0764 | 0.0680 | 0.0463 | 0.8186 |
| DDBC-release, trained | 0.0669 | 0.0777 | 0.0688 | 0.0463 | 1323 |
| DDBC-paper | 0.1087 | 0.1325 | 0.1091 | 0.0747 | 1336 |

**Setting:** the released `pog_dense` data (20,780 training bundles, 2,521 test queries, 31,217 items), k = 4 with 2 given items, `seed=1`, one Kaggle GPU. DDBC-release uses the released IDs. The example gives the 256-step checkpoint row with `DDBC_CKPT` set. The 25-step row and the trained row come from separate runs with the same split and settings. The release recipe decodes with 256 steps, the default of the released `configs/config.yaml`, while the released checkpoints store `sampling.steps: 25`, the value of the paper. Over seeds 1 to 3, the 25-step checkpoint gives Set-F1 0.0680, 0.0656 and 0.0668.

The released Spotify checkpoints, with the same evaluation:

| k | `sampling_steps` | NDCG@20 | Recall@20 | Set-F1 | Set-Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30 | 256 | 0.0150 | 0.0172 | 0.0128 | 0.0067 |
| 30 | 25 | 0.0159 | 0.0183 | 0.0136 | 0.0071 |
| 60 | 256 | 0.0226 | 0.0156 | 0.0226 | 0.0118 |
| 60 | 25 | 0.0237 | 0.0154 | 0.0230 | 0.0120 |
| 90 | 256 | 0.0317 | 0.0141 | 0.0308 | 0.0162 |

**A few things to note:**

- In this run, the co-occurrence baseline beats both DDBC recipes on every metric on POG.
- The paper recipe was not trained on the dataset Spotify.
