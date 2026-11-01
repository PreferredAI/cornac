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

The table reports CPU inference through the Cornac wrapper using six retained
checkpoints on Amazon Reviews 2014 Beauty: 12,068 items, 22,363 test queries,
beam 20 and history limit 210. Training was not rerun through this wrapper.
ST5 uses 768-dimensional Sentence-T5 features; OpenAI uses 3072-dimensional
`text-embedding-3-large` features. Values are mean ± sample SD across joint-training
seeds 2025/2026/2027 with a fixed seed-2024 initializer; metrics are fractions.

| Features | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- |
| ST5 | 0.041378 ± 0.001322 | 0.027401 ± 0.000342 | 0.063021 ± 0.001201 | 0.034355 ± 0.000252 | 0.093294 ± 0.003556 | 0.042015 ± 0.000854 |
| OpenAI | 0.044135 ± 0.001538 | 0.028961 ± 0.000630 | 0.069236 ± 0.001704 | 0.037026 ± 0.000674 | 0.105338 ± 0.001701 | 0.046100 ± 0.000367 |

The retained features, initializers and checkpoints are not distributed with the
package; the presets alone do not reproduce this evaluation.

### TIGER and RPG reference results

For context, the following Beauty results are reported in the
[TIGER](../tiger/README.md#best-tiger-configs-vs-tuned-sasrec-beauty) and
[RPG](../rpg/README.md#graph-versus-exact-scoring) model documentation:

| Model / recipe | Features | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TIGER / `PAISCHER_CONFIG`, RQVAE | ST5-base | 0.0419 | 0.0270 | 0.0634 | 0.0338 | 0.0953 | 0.0419 |
| TIGER / `PAISCHER_CONFIG`, rkmeans | OpenAI | 0.0405 | 0.0265 | 0.0678 | 0.0353 | 0.1045 | 0.0445 |
| RPG / `RPG_BEAUTY_CONFIG`, graph | OpenAI | 0.0533 | 0.0372 | 0.0789 | 0.0454 | 0.1113 | 0.0536 |

These reference runs use seed 123; DIGER reports three-seed means above.
TIGER uses history length 20 and constrained beam 50; RPG uses history length
50 and graph beam 20, versus DIGER's history limit 210 and unconstrained beam
20. All use Amazon Reviews 2014 Beauty, but identical filtering, feature arrays
and evaluation populations have not been established. The rows provide context
rather than a controlled comparison.

## Semantic ID evaluation

Set `semantic_id_metrics=True` on `NextItemEvaluation`, or add
`--semantic-id-evaluation` to the example, to report SID and CCE Recall/NDCG
alongside ordinary item metrics from one inference. DIGER's complete semantic
IDs include a deterministic collision suffix, so each collision class contains
one item and all three metric views coincide in the reported runs.

Invalid and repeated generated IDs are compacted for semantic-ID metrics. The
evaluator warns about short beams without expanding them. See the
[Semantic ID guide](../../../docs/source/api_ref/eval_methods.rst).
