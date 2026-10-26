# RPG

Cornac implementation of **RPG** (Generating Long Semantic IDs in Parallel for
Recommendation, Hou et al., KDD 2025,
[arXiv:2506.05781](https://arxiv.org/abs/2506.05781)). RPG represents each item
with a long, unordered Semantic ID produced by product quantization and predicts
all of the next item's Semantic-ID digits in parallel. This port follows the
[official implementation](https://github.com/facebookresearch/RPG_KDD2025) for
OPQ tokenization, causal-session training, the GPT-2 multi-token-prediction
(MTP) backbone, and similarity-graph-guided decoding.

## Requirements

RPG needs PyTorch, Transformers, FAISS, and scikit-learn:

```bash
pip install -r cornac/models/rpg/requirements.txt
```

The FAISS tokenizer uses 8-bit product quantization, so the supported
`codebook_size` is 256. Item content embeddings must cover every item known to
the train, validation, and test splits.

## Usage

Supply precomputed item content embeddings through a `FeatureModality`. Its
`features` rows must align with `ids` (see
[`examples/rpg_example.py`](../../../examples/rpg_example.py) for an end-to-end
example):

```python
from cornac.data import FeatureModality
from cornac.eval_methods import NextItemEvaluation
from cornac.models import RPG
from cornac.models.rpg import RPG_BEAUTY_CONFIG

eval_method = NextItemEvaluation.from_splits(
    semantic_id_metrics=True,
    train_data=train,
    val_data=val,
    test_data=test,
    mode="last",
    item_feature=FeatureModality(features=item_embeddings, ids=item_ids),
)

model = RPG(**{**RPG_BEAUTY_CONFIG, "device": "auto", "seed": 2024})
```

The paper uses 3072-dimensional OpenAI `text-embedding-3-large` item
embeddings, followed by whitened PCA to 512 dimensions. Other content
embeddings are accepted, but Semantic-ID quality and recommendation results
depend on them.

Constructor defaults provide a lighter training setup. Use `RPG_CONFIG` for
the official repository defaults or a per-dataset configuration for the
paper-best Amazon recipe:

| Config              |     LR | Temperature | Codebooks | Beams | Graph edges | Propagation steps |
| ------------------- | -----: | ----------: | --------: | ----: | ----------: | ----------------: |
| `RPG_CONFIG`        | 0.0003 |        0.07 |        32 |    50 |          50 |                 3 |
| `RPG_BEAUTY_CONFIG` |   0.01 |        0.03 |        32 |    20 |         200 |                 3 |
| `RPG_SPORTS_CONFIG` |  0.003 |        0.03 |        16 |   100 |          30 |                 5 |
| `RPG_TOYS_CONFIG`   |  0.003 |        0.03 |        16 |   200 |          20 |                 3 |

All shipped configurations use a maximum history length of 50, batch size 256,
AdamW, cosine scheduling with 10,000 warmup steps, and at most 150 epochs. The
validation NDCG@10 is evaluated every epoch in batches of 32; training stops
after 20 consecutive non-improving epochs and restores the best checkpoint.

## Training iteration

RPG does **not** expand a short session into independent prefix-target rows as
TIGER, LETTER, and SASRec do. For a session `[a, b, c, d]`, RPG creates one
causal training row:

```text
input:  [a, b, c, PAD, ...]
labels: [b, c, d, -100, ...]
```

The GPT-2 backbone produces a hidden state at every input position. Each valid
label contributes an MTP loss that predicts all Semantic-ID codebooks in
parallel; `-100` labels are ignored. Consequently, the three next-item targets
share one forward pass and one optimizer step.

For a session longer than `max_len + 1`, the first window supervises every
position. Later sliding windows supervise only their final position, ensuring
that every next-item target is counted exactly once. A session of length `T`
therefore produces `max(T - max_len, 1)` training rows, rather than `T - 1`
prefix rows.

## Scoring modes

- `scoring="graph"` follows the paper: it propagates a beam over a similarity
  graph built from item Semantic IDs and assigns real scores only to the final
  candidates. Set `n_beams` at least as large as the largest evaluation cutoff.
- `scoring="exact"` scores the full catalog by gathering and averaging the
  parallel per-codebook logits. It is useful as a deterministic diagnostic and
  for modest catalogs, but it is not the paper's retrieval path.

Both modes use the same trained model. On the Amazon experiments below, their
Recall/NDCG results differ by at most 0.0018 in absolute value.

## Results

Paper-style graph-decoding results on the three Amazon Reviews 2014 5-core
datasets:

| Dataset |  R@5 ours/paper |  N@5 ours/paper | R@10 ours/paper | N@10 ours/paper |
| ------- | --------------: | --------------: | --------------: | --------------: |
| Beauty  | 0.0533 / 0.0550 | 0.0372 / 0.0381 | 0.0789 / 0.0809 | 0.0454 / 0.0464 |
| Sports  | 0.0288 / 0.0314 | 0.0198 / 0.0216 | 0.0420 / 0.0463 | 0.0241 / 0.0263 |
| Toys    | 0.0613 / 0.0592 | 0.0415 / 0.0401 | 0.0898 / 0.0869 | 0.0506 / 0.0490 |

**Setting:** per-user leave-last-out (`train = sequence[:-2]`, validation target
= second-to-last, test target = last), `mode="last"`, `max_len=50`, and
`seed=123`. Items use the official RPG `text-embedding-3-large` embeddings.
Each category uses its per-dataset configuration above. Beauty and Sports are
within 3.1% and 8.3% of the paper's R@5, respectively; Toys is 3.5% higher.

### Graph versus exact scoring

| Dataset | Scoring |    R@5 |    N@5 |   R@10 |   N@10 |   R@20 |   N@20 |    MRR | Total (s) |
| ------- | ------- | -----: | -----: | -----: | -----: | -----: | -----: | -----: | --------: |
| Beauty  | graph   | 0.0533 | 0.0372 | 0.0789 | 0.0454 | 0.1113 | 0.0536 | 0.0380 |      2502 |
| Beauty  | exact   | 0.0534 | 0.0373 | 0.0798 | 0.0457 | 0.1125 | 0.0539 | 0.0412 |         — |
| Sports  | graph   | 0.0288 | 0.0198 | 0.0420 | 0.0241 | 0.0619 | 0.0291 | 0.0219 |      4095 |
| Sports  | exact   | 0.0291 | 0.0201 | 0.0428 | 0.0245 | 0.0627 | 0.0295 | 0.0229 |         — |
| Toys    | graph   | 0.0613 | 0.0415 | 0.0898 | 0.0506 | 0.1233 | 0.0591 | 0.0439 |      1682 |
| Toys    | exact   | 0.0611 | 0.0414 | 0.0903 | 0.0508 | 0.1251 | 0.0596 | 0.0444 |         — |

## Practical notes

- `n_codebook` controls Semantic-ID length. The paper-best Sports and Toys
  recipes use 16 codebooks, Beauty uses 32, and the paper's long-ID
  CDs-and-Vinyl experiment uses 64.
- OPQ is trained only on items present in the training interactions, while the
  resulting tokenizer assigns codes to all items with supplied features.
- Content embeddings are part of the model recipe. Random features are useful
  for smoke tests but do not provide meaningful semantic IDs.
- Graph decoding is approximate and stochastic because its initial beam is
  sampled. Set `seed` for repeatability, or use exact scoring when deterministic
  full-catalog ranks are required.


### Semantic ID evaluation

Set `NextItemEvaluation(..., semantic_id_metrics=True)` to report ordinary
item metrics plus `SID-Recall@K`, `SID-NDCG@K`, `CCE-Recall@K` and `CCE-NDCG@K` for
requested finite-K Recall/NDCG metrics. All views share one inference; CCE averages
item credit over uniform order within each admitted collision bucket. Invalid
and repeated predicted IDs are compacted. Other requested metrics keep ordinary
item evaluation. Configured beam sizes remain unchanged; short candidate lists
warn. Set `semantic_id_metrics=False` for ordinary item metrics only.

See the [evaluation documentation](../../../docs/source/api_ref/eval_methods.rst)
and [mixed-model example](../../../examples/semantic_id_evaluation_example.py).
