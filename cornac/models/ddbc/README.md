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
| Spotify | `DDBC_PAPER_SPOTIFY{30,60,90}_CONFIG` | `DDBC_RELEASE_SPOTIFY{30,60,90}_CONFIG` | 30 / 60 / 90 | 128 paper, 256 release | 0.4 |

The paper recipe uses C = 128 on both datasets, as the review of this PR asks. We could not find that value in the paper, which gives every level the same size C without stating C. The release recipe follows the released files: 128 on POG, the size of the released `clhe_weight.npy`, and 256 on Spotify, the codebook size in the released config.

The paper numbers the dedup level, the collision digit, within the codebook size of that level, so it also takes `codebook_size` values. When more items share the codes of every level, DDBC gives the collision digit as many values as the largest such group and warns. In our runs this happened only for the paper recipe on Spotify, where 333 items share one ID at C = 128 on a Kaggle T4 GPU with `seed=1` (the IDs differ between CPU and GPU). It is the one place where the paper recipe cannot follow the paper literally. On POG and with `item_sids` the collision digit keeps `codebook_size` values, so the released checkpoints load unchanged.

The recipes differ in these arguments:

| Argument | Paper recipe | Release recipe |
| --- | --- | --- |
| `tokenizer` | `"rvq"`: residual quantizer with greedy encoding, codebooks trained by gradient descent | `"faiss"`: Faiss `IndexResidualQuantizer`, inner-product metric, beam search |
| `fallback_levels` | 3: an ID that names no item is matched through the sum of its codewords | 2: levels 1 and 2 only |
| `layout` | `"joint"`: given items and masked slots in one `<bos> ... <eos>` sequence | `"split"`: `<bos>` given `<eos>` `<bos>` slots `<eos>` |
| `n_diffusion_steps` | 25: discrete t in {1, ..., T}, mask rate t / T | `None`: continuous t in [0.001, 1] |
| `time_conditioning` | `True`: the denoiser sees t | `False`: it sees 0, as the release evaluates every checkpoint. The Spotify k = 30 checkpoint was trained with it on |
| `decode` | `"confidence"`: most confident tokens first | `"ancestral"`: the release's sampler |
| `sampling_steps` | 25 | 256 |

Both recipes share the paper's DiT (6 blocks, hidden size 64, 8 heads) with the release's conditioning size 128 and dropout 0.1, the paper's 20,000 optimizer steps, and the released code's training settings, which the paper does not state: AdamW at 3e-4 with 500 warm-up steps and then a constant rate, clipping 1.0, EMA decay 0.9999, validation every 32 batches, and batch 300 with 2 accumulation steps. Both keep the averaged weights with the lowest validation loss. Both also inherit four behaviours of the release: the code embeddings start from their codewords, the times of a batch are drawn antithetically, the weight decay is 0, and the loss is the batch sum divided by the sequence length.

The `"rvq"` tokenizer starts from residual k-means and trains its codebooks with full-batch Adam. The paper gives none of these settings. The paper also uses a straight-through estimator, which only matters when gradients reach an encoder. Here the item features are fixed, so the reconstruction term trains the codebooks directly. Faiss training is random, so `tokenizer="faiss"` does not give the released IDs. To use the released checkpoints, pass the released `clhe_token.json` as `item_sids` and `clhe_weight.npy` as `codebooks`, as the example does.

## Implementation notes

- **Loss.** Each masked code token costs its cross-entropy times `T / t` in discrete time and `1 / t` in continuous time, the MDLM weights. The paper prints the cross-entropy without a weight.
- **Decoding.** A code position may only take codes of its own level. `"confidence"` reveals at each step the masked positions whose best token is most probable, `ceil(remaining / steps left)` of them. The paper leaves the reveal set open, so this count is our choice. `"ancestral"` follows the release's sampler, which draws each masked position with Gumbel-max.
- **`construct`** turns each generated ID into a vector, the vector of the item with that ID or, for an ID that names no item, the sum of its first `fallback_levels` codewords, and takes the free catalog item closest by cosine. The paper does not say how a vector is mapped to an item. Cosine follows the release. `Set-F1` and `Set-Jaccard` use these items.
- **`score`** ranks from the same decoded bundle: the constructed items first, then every other item by its highest cosine similarity to the generated slots. It decodes `bundle_size - len(given)` slots, so held-out bundles should have `bundle_size` items, as in the release.
- **Released checkpoints.** `pretrained` reads a released `.ckpt` without running code from it, loads its EMA weights and skips training. It requires `item_sids` and `codebooks`.
- **Sampling.** The decoding generator is seeded from `seed` and the query, so a query gets the same bundle whatever the evaluation order.

## Main differences vs the reference code

- **Evaluation.** `BundleConstructionEvaluation` ranks and constructs over the whole catalog, so its numbers are not comparable with those of the paper. The release scores each test bundle on a candidate pool of the targets plus 99 random items per target, which can include the given items.
- **Held-out bundles.** As in the released test files, held-out bundles are cut to their first k items, and the first ceil(k / 2) are given. With random windows of k items instead, BundlePop-cooccurrence reaches Set-F1 0.1843 on POG against 0.1656 here.
- **Training length and batch.** Both recipes train 20,000 steps on batches of 300 with 2 accumulation steps. Every released checkpoint stores a batch of 300 per GPU. Its `global_batch_size` (512, or 256 for Spotify30) only sets the accumulation, `ceil(global_batch_size / (GPUs x 300))`, and the number of GPUs is stored as `${device_count:}`, so the checkpoints do not record it. On one GPU, as in the released training script, POG, Spotify60 and Spotify90 train on 300 x 2 = 600 windows per step and Spotify30 on 300 x 1 = 300, so the release recipe's `accumulate=2` departs from the Spotify30 checkpoint. On the paper's four GPUs, every checkpoint would accumulate once. The POG, Spotify30 and Spotify60 checkpoints store `max_steps: 30000` and were saved at steps 10,375, 21,312 and 14,431. Spotify90 stores `max_steps: 1000000` and was saved at step 41,607. `max_steps`, `batch_size` and `accumulate` remain arguments.
- **Precision.** The release trains and decodes in bfloat16, with autocast around the DiT blocks. The recipes keep `use_bf16=False` so that they run on any GPU. `use_bf16=True` autocasts the same blocks in training and decoding.
- **Validation masks.** The validation loss draws its masks with a fixed seed, so checkpoints are compared on the same masks. The release draws new masks at every validation.

## Differences neither recipe can express

No argument covers these, and both recipes behave the same way:

- The code embeddings always start from their codewords when `hidden_size` equals the feature size.
- The collision digit grows beyond `codebook_size` values when more items share one ID, where the release raises an error.
- The paper prints the masked cross-entropy without a per-step weight. Both recipes weight it by `T / t` or `1 / t` (see Implementation notes).
- The release sampler lets a slot code position draw `<bos>`. DDBC allows only codes of the position's level. Allowing `<bos>` gave identical tokens on the released POG and Spotify k = 30 checkpoints.
- The validation masks are fixed here and drawn anew in the release (see above).

## Results

Test results of [`examples/ddbc_pog.py`](../../../examples/ddbc_pog.py) on POG:

| Model | NDCG@20 | Recall@20 | Set-F1 | Set-Jaccard | Train(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| BundlePop-item | 0.1445 | 0.2237 | 0.0841 | 0.0561 | 0.0234 |
| BundlePop-cooccurrence | **0.2472** | **0.3167** | **0.1656** | **0.1164** | 0.0055 |
| DDBC-release, released checkpoint, `sampling_steps=256` | 0.0683 | 0.0817 | 0.0706 | 0.0477 | 0.3190 |
| DDBC-release, released checkpoint, `sampling_steps=25` | 0.0646 | 0.0764 | 0.0680 | 0.0463 | 0.2518 |
| DDBC-release, trained | 0.0669 | 0.0777 | 0.0688 | 0.0463 | 1349 |
| DDBC-paper | 0.1105 | 0.1323 | 0.1085 | 0.0739 | 1449 |

**Setting:** the released `pog_dense` data (20,780 training bundles, 2,521 test queries, 31,217 items), k = 4 with 2 given items, `seed=1`, one Kaggle GPU. DDBC-release uses the released IDs. The example gives the 256-step checkpoint row with `DDBC_CKPT` set. The BundlePop, 256-step checkpoint and DDBC-paper rows come from one run. The 25-step row and the trained DDBC-release row come from a second run on the same code, which gave the 256-step row again with the same numbers. The release recipe decodes with 256 steps, the default of the released config, which the released evaluation script does not override. The `sampling.steps: 25` stored in the checkpoints comes from the released training script and only feeds the samples logged at validation. The paper decodes with T = 25, which the paper recipe follows. Over seeds 1 to 3, the 25-step checkpoint gives Set-F1 0.0680, 0.0656 and 0.0668. BundlePop counts every bundle of `train.txt`, the rows DDBC cuts its training windows from.

**Reproducibility.** Decoding a released checkpoint twice gave identical numbers. Two trainings of the paper recipe with seed 1 on a Kaggle T4 GPU gave Set-F1 0.1059 and 0.1091, because the `"rvq"` tokenizer gave different IDs on each run. Its residual k-means summed the centroids in a nondeterministic order on CUDA, and the gradient of the codeword lookup was also summed in a nondeterministic order: with indexing on CPU, and with `F.embedding` over 500 Adam steps on a T4. The k-means now runs on CPU and the lookup multiplies a one-hot matrix by the codebook. On a Kaggle T4, the paper recipe trained twice with seed 1 then gave the same IDs, identical weights and the DDBC-paper row above both times, and an earlier session on the same GPU type gave that row too. CPU and GPU still give different IDs.

The released Spotify checkpoints, with the same evaluation:

| k | `sampling_steps` | NDCG@20 | Recall@20 | Set-F1 | Set-Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30 | 256 | 0.0150 | 0.0172 | 0.0128 | 0.0067 |
| 30 | 25 | 0.0159 | 0.0183 | 0.0136 | 0.0071 |
| 60 | 256 | 0.0226 | 0.0156 | 0.0226 | 0.0118 |
| 60 | 25 | 0.0237 | 0.0154 | 0.0230 | 0.0120 |
| 90 | 256 | 0.0317 | 0.0141 | 0.0308 | 0.0162 |

**A few things to note:**

- The co-occurrence baseline beats both DDBC recipes on every metric on POG.
- The paper recipe was trained on Spotify k = 30 only for 300 steps, to check that it runs at C = 128 and C = 256. Its test Set-F1 was 0.0103 and 0.0114, which is not comparable with the released checkpoints.
