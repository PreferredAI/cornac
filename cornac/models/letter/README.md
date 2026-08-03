# LETTER

Cornac implementation of **LETTER** (Learnable Item Tokenization for Generative Recommendation, Wang et al., CIKM 2024, [arXiv:2405.07314](https://arxiv.org/abs/2405.07314)). LETTER adds collaborative alignment and code-assignment diversity to an RQ-VAE tokenizer, then trains a T5 generator to predict the next item's four-token Semantic ID. This implementation follows the [released code](https://github.com/HonghuiBao2000/LETTER) for both stages.

## Requirements

Install the optional PyTorch, Transformers, and constrained-k-means dependencies:

```bash
pip install -r cornac/models/letter/requirements.txt
```

LETTER needs two aligned feature matrices covering every item known to the train, validation, and test splits:

- item content embeddings, supplied through `FeatureModality`; and
- 32-dimensional collaborative item embeddings and their raw item IDs,
  supplied through `cf_embeddings` and `cf_embedding_ids` (the paper uses
  SASRec item embeddings). LETTER remaps these rows to Cornac's global item
  indices during fitting.

## Usage

```python
from cornac.data import FeatureModality
from cornac.eval_methods import NextItemEvaluation
from cornac.models import LETTER
from cornac.models.letter import LETTER_BEAUTY_CONFIG

eval_method = NextItemEvaluation.from_splits(
    train_data=train,
    val_data=val,
    test_data=test,
    mode="last",
    item_feature=FeatureModality(features=item_embeddings, ids=item_ids),
)

model = LETTER(
    **{
        **LETTER_BEAUTY_CONFIG,
        "cf_embeddings": sasrec_item_embeddings_32d,
        "cf_embedding_ids": sasrec_item_ids,
        "device": "auto",
        "seed": 42,
    }
)
```

See [`examples/letter_example.py`](../../../examples/letter_example.py) for a small two-stage API example with stand-in features. `LETTER_BEAUTY_CONFIG` is the reproduction recipe; `LETTER_CONFIG` keeps the paper-wide recommended regularization weights.

## Training

LETTER is trained in two stages. The tokenizer is a four-level RQ-VAE with collaborative alignment, code-assignment diversity, and collision handling. The generator is a T5 model that predicts the four-token Semantic ID of the next item. `LETTER_BEAUTY_CONFIG` contains the released Beauty training settings.

The released “ranking-guided” objective uses a temperature of 1.0, making it equivalent to ordinary token cross-entropy. Generator AdamW weight decay follows Hugging Face Trainer: layer-normalization and bias parameters are placed in a zero-decay group.

`precomputed_semantic_ids` is a positional table whose rows must already follow Cornac's global item-index order. If semantic IDs come with raw item IDs, remap them through the evaluation method's `global_iid_map` before constructing `LETTER`.

## Beauty reproduction

All results use the Amazon Beauty 2014 5-core interactions, seed 42, and a chronological leave-last-out split. The Cornac results have two scopes:

- **Generator + author IDs** skips tokenizer training and evaluates the Cornac generator with the authors' Semantic IDs over all 12,101 released items.
- **End-to-end** trains both stages using Sentence-T5-base title+description embeddings and locally trained 32-dimensional SASRec embeddings. It covers the 12,068-item universe for which both inputs are available.

The complete [Beauty generator example](beauty_example.py) consumes the authors' released `Beauty.index.json` and `Beauty.inter.json`, then trains once and evaluates multiple beam widths from the same checkpoint:

```bash
python -m cornac.models.letter.beauty_example Beauty.index.json \
  --interaction-file Beauty.inter.json --beams 20 50
```

Omit `--interaction-file` to use Cornac's Amazon loader with an ASIN-keyed Semantic-ID file. For a numeric-keyed index in that mode, pass a numeric-to-ASIN JSON mapping through `--item-id-map`.

### Results

Recall is abbreviated as R and NDCG as N.

| System                        | Beams |    R@5 |    N@5 |   R@10 |   N@10 |
| ----------------------------- | ----: | -----: | -----: | -----: | -----: |
| LETTER paper                  |    20 | 0.0431 | 0.0286 | 0.0672 | 0.0364 |
| Released code + author IDs    |    20 | 0.0413 | 0.0268 | 0.0645 | 0.0343 |
| Cornac generator + author IDs |    20 | 0.0420 | 0.0279 | 0.0656 | 0.0354 |
| Cornac generator + author IDs |    50 | 0.0429 | 0.0282 | 0.0670 | 0.0360 |
| Cornac end-to-end             |    50 | 0.0356 | 0.0241 | 0.0560 | 0.0307 |

### Interpretation

Across the four reported metrics, it is within `0.0016` of the paper and `0.0011` of the released-code rerun, supporting generator fidelity.

The end-to-end run trained the 10,000-epoch LETTER tokenizer and generator with reproducible substitute inputs, reducing Semantic-ID collisions from 74 to 2. It used 50 beams and predates the corrected generator optimizer, so it is retained as evidence of the full local training pipeline rather than as a controlled generator comparison.

The authors' Beauty index contains 12,101 items but only 12,088 unique Semantic IDs. The released evaluator deduplicates these candidate strings, whereas Cornac retains the raw items and gives items sharing an ID the same score; metrics involving those collisions are therefore not directly comparable.

### Reproduction limitation

The released repository provides the final Beauty Semantic-ID table, but not the trained tokenizer checkpoint, content embeddings, or SASRec checkpoint used to produce it. Exact reproduction of the paper's learned Semantic IDs therefore requires those missing artifacts; the end-to-end result above is a runnable substitute rather than an exact reconstruction of the paper's tokenizer inputs.
