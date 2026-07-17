# TIGER

Cornac implementation of **TIGER** (Recommender Systems with Generative Retrieval, Rajput et al., NeurIPS 2023, [arXiv:2305.05065](https://arxiv.org/pdf/2305.05065)): a tokenizer (`rqvae` or `rkmeans`) quantizes item *content* embeddings into hierarchical semantic IDs, and a small T5-style seq2seq model generates the next item's IDs from the session history. There is no official code release; this port is cross-checked against [`GRID`](https://github.com/snap-research/GRID) ([arXiv:2507.22224](https://arxiv.org/abs/2507.22224)) and [`PAISCHER/Mender`](https://github.com/facebookresearch/preference_discerning) ([arXiv:2412.08604](https://arxiv.org/abs/2412.08604)).

## Usage

TIGER needs item content embeddings, supplied as a `FeatureModality` whose rows are aligned with the global item IDs (see [`examples/tiger_example.py`](../../../examples/tiger_example.py) for an end-to-end run):

```python
from cornac.data import FeatureModality
from cornac.eval_methods import NextItemEvaluation
from cornac.models import TIGER
from cornac.models.tiger import GRID_CONFIG, PAISCHER_CONFIG

eval_method = NextItemEvaluation.from_splits(
    train_data=train, test_data=test, val_data=val, mode="last",
    item_feature=FeatureModality(features=item_embeddings, ids=item_ids),
)

model = TIGER(**{**PAISCHER_CONFIG, "n_beams": 50, "seed": 123})
```

Two recipes ship with the package: `GRID_CONFIG` prioritizes speed (rkmeans tokenizer, no gradient training, short budget - roughly 1/3 -- 1/2 the training time at 78-86% of the Recall@5), while `PAISCHER_CONFIG` follows the original paper closely for the best results. The constructor defaults reproduce the paper's *architecture* but not its training recipe and reach only ~half of `PAISCHER_CONFIG`'s accuracy.

GRID uses a single config for all datasets; PAISCHER tunes per dataset, and are shipped as `PAISCHER_SPORTS_CONFIG` (lr 1e-4, batch 256, beam 10) and `PAISCHER_TOYS_CONFIG` (d_model 196, d_ff 1536, half the training budget, beam 10).

## Main differences vs GRID / PAISCHER

- **Epoch loop, not step loop.** GRID/PAISCHER train by optimizer *steps* with step-based early stopping (GRID: validate every 100 steps, patience 10; PAISCHER: 200k-step cap, patience 15). Cornac trains `n_epochs` and validates every `val_eval_every` *epochs*, keeping the best checkpoint (`model_selection="best"`) - coarser model selection (one beauty epoch at batch 64 ~= 2k steps) and no early exit, so the full budget always runs. The shipped configs convert their step budgets to epochs (`n_steps = n_epochs × ceil(n_samples / batch_size)`); in practice epoch-level best-on-val was close enough to match their published numbers.
- **No user-ID token**: the paper hashes users into 2000 buckets; GRID's ablation finds removing it optimal, so it is omitted here.
- **Both scoring modes from one model**: constrained beam search *and* exact full-catalog log-likelihood ranking - the latter is how beam ~= exact was verified.

## Results

`Recall@5` across the three Amazon-2014 5-core datasets, comparing to the TIGER paper and the GRID benchmark:

| Category | Defaults | `GRID_CONFIG` | `PAISCHER_CONFIG` | TIGER Paper | GRID published |
| -------- | -------: | ------------: | ----------------: | ----------: | -------------: |
| beauty   |   0.0217 |        0.0340 |          *0.0419* |      0.0454 |         0.0422 |
| sports   |   0.0157 |        0.0188 |          *0.0240* |      0.0264 |         0.0236 |
| toys     |   0.0192 |        0.0298 |  *0.0362* (+desc) |      0.0521 |         0.0376 |

**Setting:** `leave_last_out` split (identical to the original paper), `mode="last"`, `max_len=20`, `seed=123`. Sentence-T5-base (768-d) item embeddings over TIGER-style `Title/Price/Brand/Categories` text (`amazon_review.load_text`); `(+desc)` appends item descriptions. TIGER scored with constrained beam search, `n_beams=50`.

### Detailed tuning, base encoder

| Category | Recipe         | Tokenizer |    R@5 |    N@5 |   R@10 |   N@10 |   R@20 |   N@20 | Train(s) |
| -------- | -------------- | --------- | -----: | -----: | -----: | -----: | -----: | -----: | -------: |
| beauty   | default        | rqvae     | 0.0217 | 0.0132 | 0.0386 | 0.0186 | 0.0591 | 0.0238 |     8072 |
| beauty   | grid           | rkmeans   | 0.0340 | 0.0216 | 0.0564 | 0.0288 | 0.0885 | 0.0368 |     4772 |
| beauty   | paischer       | rqvae     | 0.0419 | 0.0270 | 0.0634 | 0.0338 | 0.0953 | 0.0419 |     4341 |
| beauty   | paischer       | rkmeans   | 0.0404 | 0.0259 | 0.0641 | 0.0335 | 0.1005 | 0.0427 |     9386 |
| sports   | default        | rqvae     | 0.0157 | 0.0098 | 0.0262 | 0.0132 | 0.0442 | 0.0177 |    11075 |
| sports   | grid           | rkmeans   | 0.0188 | 0.0117 | 0.0329 | 0.0162 | 0.0540 | 0.0215 |     5503 |
| sports   | paischer       | rqvae     | 0.0230 | 0.0149 | 0.0373 | 0.0195 | 0.0596 | 0.0251 |    14102 |
| sports   | paischer       | rkmeans   | 0.0240 | 0.0155 | 0.0394 | 0.0204 | 0.0635 | 0.0265 |    15917 |
| toys     | default        | rqvae     | 0.0192 | 0.0115 | 0.0325 | 0.0158 | 0.0509 | 0.0204 |     6366 |
| toys     | grid           | rkmeans   | 0.0298 | 0.0189 | 0.0506 | 0.0256 | 0.0780 | 0.0325 |     4270 |
| toys     | paischer       | rqvae     | 0.0320 | 0.0205 | 0.0521 | 0.0269 | 0.0822 | 0.0345 |     9698 |
| toys     | paischer       | rkmeans   | 0.0348 | 0.0222 | 0.0578 | 0.0295 | 0.0922 | 0.0382 |     9544 |
| toys     | paischer +desc | rkmeans   | 0.0362 | 0.0237 | 0.0583 | 0.0308 | 0.0915 | 0.0391 |     8609 |

### Best TIGER configs vs tuned SASRec (beauty)

SASRec tuned over loss {`ce`, `bce`, `bpr`, `bpr-max`} × emb {64, 128} × lr {5e-4, 1e-3, 5e-3} × dropout {0.2, 0.5} × `max_len` {20, 50}, best by validation NDCG@10 (`bce`, 128 / 5e-4 / 0.2 / 50). TIGER rows are `PAISCHER_CONFIG` + beam scoring:

| Model                     |        R@5 |        N@5 |       R@10 |       N@10 |       R@20 |       N@20 | Train(s) |
| ------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: | -------: |
| TIGER (rqvae / ST5-base)  |     0.0419 |     0.0270 |     0.0634 |     0.0338 |     0.0953 |     0.0419 |     4341 |
| TIGER (rkmeans / ST5-XXL) |     0.0419 |     0.0266 |     0.0660 |     0.0344 |     0.1016 |     0.0434 |     4247 |
| TIGER (rkmeans / openai)  |     0.0405 |     0.0265 |     0.0678 |     0.0353 | **0.1045** |     0.0445 |    10013 |
| SASRec (tuned, bce)       | **0.0508** | **0.0348** | **0.0740** | **0.0423** |     0.1037 | **0.0498** |     2455 |

**A few things to note:**

- `TIGER` performance can fluctuate with different training *recipe*: `PAISCHER_CONFIG` (6+6 layers, lr 3e-4 cosine + 10k warmup, dropout 0.2, best-on-val selection) roughly doubles the defaults with no architectural change.
- `tokenizer="rkmeans"` matches or beats `rqvae` (within ~4% per category) at zero tokenizer-training cost (GRID's ablation studies).
- `beam` scoring ~= `exact` scoring in every run (<=0.5%); need to set `n_beams >=` largest metric cutoff @K.
- `TIGER` likely uses `Sentence-T5-XXL` embeddings, but improvements over ST5-base are minimal (sometimes even decrease). OpenAI's `text-embedding-3-large` (provided by [`RPG`](https://github.com/facebookresearch/RPG_KDD2025)) is also available, but it doesn't guarantee performance improvement.
- `beauty` and `sports` are somewhat reproducible, but `toys` is ***not***, likely due to ambiguity in the item text used (possibly `descriptions` only).
- Well-tuned `SASRec` and `BERT4Rec` (within cornac) still outperform `TIGER`: tuned SASRec on beauty wins every metric except a marginal R@20 edge for `TIGER` + openai embedding, at a fraction of the cost (pure ID-based recommendation).
