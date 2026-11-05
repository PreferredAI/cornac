# DIGER

Cornac implementation of **DIGER** (Differentiable Semantic ID for Generative
Recommendation, Fu et al., SIGIR 2026,
[arXiv:2601.19711](https://arxiv.org/abs/2601.19711)). A residual quantizer turns
item content embeddings into semantic IDs; a T5 generator predicts the next
item's ID. Frequency-gated Gumbel assignments let recommendation gradients
update the codebooks alongside the generator.

## Requirements

```bash
pip install -r cornac/models/diger/requirements.txt
```

## Usage

Run the [Amazon Beauty example](../../../examples/diger_example.py) with
precomputed Sentence-T5 embeddings:

```bash
python examples/diger_example.py --features beauty-st5.npz
```

The NPZ archive must contain `features` (a finite float32 matrix) and `ids`
(raw item IDs aligned with its rows), covering every item in the evaluation
splits. Add `--embedding openai` for precomputed OpenAI embeddings; no API key is
needed. Without `--features`, the example generates Sentence-T5 embeddings from
Cornac's item texts (requires `sentence-transformers`). These regenerated
features may differ from those used for the results below.

Supply embeddings through `FeatureModality` in the Python API:

```python
from cornac.data import FeatureModality
from cornac.eval_methods import NextItemEvaluation
from cornac.models import DIGER
from cornac.models.diger import BEAUTY_ST5_CONFIG

eval_method = NextItemEvaluation.from_splits(
    train_data=train,
    val_data=val,
    test_data=test,
    mode="last",
    item_feature=FeatureModality(features=item_embeddings, ids=item_ids),
)
model = DIGER(**{**BEAUTY_ST5_CONFIG, "device": "auto", "seed": 2025})
```

`BEAUTY_ST5_CONFIG` and `BEAUTY_OPENAI_CONFIG` provide the selected Beauty
training recipe. Validation NDCG@10 is evaluated every epoch; training stops
after 15 non-improving epochs and restores the paired generator, tokenizer and
catalog.

## Implementation notes

- Training uses every history/next-item prefix; validation uses each session's
  final target.
- A fresh fit pretrains a feature-width-matched RQ autoencoder (20 warmup epochs,
  residual Lloyd initialization, 100 reconstruction/VQ epochs), then freezes its
  encoder. This initialization adapts the released recipe. A bundle from
  `training.pretrain_tokenizer` can be reused through `initializer` with the same
  ordered features and tokenizer options.
- Joint training updates the codebooks and generator together; full-catalog
  Sinkhorn assignments refresh each epoch. A deterministic suffix resolves
  semantic-ID collisions. Unconstrained beam search scores valid full addresses;
  set `n_beams` at least as large as the largest evaluation cutoff and no larger
  than the codebook vocabulary.

## Results

The DIGER rows report CPU inference through the Cornac wrapper using six retained
checkpoints on Amazon Reviews 2014 Beauty: 12,068 items, 22,363 test queries,
beam 20 and history limit 210. Training was not rerun through this wrapper.
ST5 uses 768-dimensional Sentence-T5 features; OpenAI uses 3072-dimensional
`text-embedding-3-large` features. Values are mean ± sample SD across joint-training
seeds 2025/2026/2027 with a fixed seed-2024 initializer; metrics are fractions.

For reference, the table also includes Beauty results from the
[TIGER](../tiger/README.md#best-tiger-configs-vs-tuned-sasrec-beauty) and
[RPG](../rpg/README.md#graph-versus-exact-scoring) documentation. DIGER rows
report three-seed mean ± sample SD; reference rows report a single run (seed 123).

| Model / recipe | Features | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DIGER | ST5 | 0.041378 ± 0.001322 | 0.027401 ± 0.000342 | 0.063021 ± 0.001201 | 0.034355 ± 0.000252 | 0.093294 ± 0.003556 | 0.042015 ± 0.000854 |
| [TIGER / `PAISCHER_CONFIG`, RQVAE](../tiger/README.md#best-tiger-configs-vs-tuned-sasrec-beauty) | ST5-base | 0.0419 | 0.0270 | 0.0634 | 0.0338 | 0.0953 | 0.0419 |
| DIGER | OpenAI | 0.044135 ± 0.001538 | 0.028961 ± 0.000630 | 0.069236 ± 0.001704 | 0.037026 ± 0.000674 | 0.105338 ± 0.001701 | 0.046100 ± 0.000367 |
| [TIGER / `PAISCHER_CONFIG`, rkmeans](../tiger/README.md#best-tiger-configs-vs-tuned-sasrec-beauty) | OpenAI | 0.0405 | 0.0265 | 0.0678 | 0.0353 | 0.1045 | 0.0445 |
| [RPG / `RPG_BEAUTY_CONFIG`, graph](../rpg/README.md#graph-versus-exact-scoring) | OpenAI | 0.0533 | 0.0372 | 0.0789 | 0.0454 | 0.1113 | 0.0536 |

TIGER uses history length 20 and constrained beam 50; RPG uses history length
50 and graph beam 20, versus DIGER's history limit 210 and unconstrained beam
20. All use Amazon Reviews 2014 Beauty, but identical filtering, feature arrays
and evaluation populations have not been established. These are reference
results rather than a controlled comparison.

The retained DIGER features, initializers and checkpoints are not distributed
with the package; the presets alone do not reproduce this evaluation.

### Tuning

For each feature setting, the presets were selected by validation NDCG@10 from
six learning-rate candidates (LR 2.5e-4/5e-4/1e-3 × ID LR 1e-5/3e-5, history
210, one seed). A wider search then trained 12 further configurations per
feature setting through this wrapper for 120 epochs with seed 2025 (history
20/50 × LR 3e-4/5e-4 × ID LR 1e-5/3e-5/1e-4), plus a wrapper retrain of the
preset. Candidates were screened on a fixed 2,000-query validation subset, and
the best two per setting were rescored on the full validation set (22,363
queries). Only the OpenAI leader beat both seed-2025 preset references there, so
it was retrained with seeds 2026 and 2027.

| Features | Recipe | Run | Validation Recall@10 | Validation NDCG@10 | Test Recall@10 | Test NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI | Preset: history 210, LR 5e-4, ID LR 3e-5 | Best seed (2026) | 0.0961 | 0.0538 | 0.0676 | 0.0363 |
| OpenAI | Preset: history 210, LR 5e-4, ID LR 3e-5 | Mean ± SD, 3 seeds | 0.0968 ± 0.0006 | 0.0533 ± 0.0004 | 0.0692 ± 0.0017 | 0.0370 ± 0.0007 |
| OpenAI | Search: history 20, LR 3e-4, ID LR 1e-4 | Best seed (2025) | 0.0977 | 0.0539 | 0.0685 | 0.0361 |
| OpenAI | Search: history 20, LR 3e-4, ID LR 1e-4 | Mean ± SD, 3 seeds | 0.0965 ± 0.0016 | 0.0525 ± 0.0014 | 0.0690 ± 0.0009 | 0.0366 ± 0.0006 |
| ST5 | Preset: history 210, LR 5e-4, ID LR 3e-5 | Best seed (2025) | 0.0890 | 0.0492 | 0.0622 | 0.0341 |
| ST5 | Preset: history 210, LR 5e-4, ID LR 3e-5 | Mean ± SD, 3 seeds | 0.0874 ± 0.0014 | 0.0481 ± 0.0010 | 0.0630 ± 0.0012 | 0.0344 ± 0.0003 |
| ST5 | Search: history 50, LR 5e-4, ID LR 1e-5 | Single seed (2025) | 0.0873 | 0.0482 | 0.0617 | 0.0331 |

The best seed here refers to the one with the highest validation NDCG@10.

## Semantic ID evaluation

Set `semantic_id_metrics=True` on `NextItemEvaluation`, or add
`--semantic-id-evaluation` to the example, to report SID and CCE Recall/NDCG
alongside ordinary item metrics from one inference. DIGER's complete semantic
IDs include a deterministic collision suffix, so each collision class contains
one item and all three metric views coincide in the reported runs.

Invalid and repeated generated IDs are compacted for semantic-ID metrics. The
evaluator warns about short beams without expanding them. See the
[Semantic ID guide](../../../docs/source/api_ref/eval_methods.rst).
