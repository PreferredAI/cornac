# LETTER

Cornac implementation of **LETTER** (Learnable Item Tokenization for Generative Recommendation, Wang et al., CIKM 2024, [arXiv:2405.07314](https://arxiv.org/abs/2405.07314)). LETTER adds collaborative alignment and code-assignment diversity to an RQ-VAE tokenizer, then trains a T5 generator to predict the next item's four-token Semantic ID. This implementation follows the [released code](https://github.com/HonghuiBao2000/LETTER) for both stages.

## Requirements

Install the optional PyTorch, Transformers, and constrained-k-means dependencies:

```bash
pip install -r cornac/models/letter/requirements.txt
```

LETTER needs two aligned feature matrices covering every item known to the train, validation, and test splits:

- item content embeddings, supplied through `FeatureModality`; and
- 32-dimensional collaborative item embeddings, supplied through `cf_embeddings` (the paper uses SASRec item embeddings).

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
        "device": "auto",
        "seed": 42,
    }
)
```

See [`examples/letter_example.py`](../../../examples/letter_example.py) for a small end-to-end example. `LETTER_BEAUTY_CONFIG` is the reproduction recipe; `LETTER_CONFIG` keeps the paper-wide recommended regularization weights.

## Training

LETTER is trained in two stages. The tokenizer is a four-level RQ-VAE with collaborative alignment, code-assignment diversity, and collision handling. The generator is a T5 model that predicts the four-token Semantic ID of the next item. `LETTER_BEAUTY_CONFIG` contains the released Beauty training settings.

The released “ranking-guided” objective uses a temperature of 1.0, making it equivalent to ordinary token cross-entropy.

## Beauty reproduction

All results use the Amazon Beauty 2014 5-core interactions, seed 42, and a chronological leave-last-out split in which the last two interactions provide the validation and test targets. Runs using the authors' precomputed Semantic IDs cover all 12,101 released items. The end-to-end Sentence-T5 run uses Cornac's standard 12,068-item known-item evaluation universe because its collaborative embeddings were trained on that universe.

### Results

Recall is abbreviated as R and NDCG as N.

| System                      |    R@5 |    N@5 |   R@10 |   N@10 |
| --------------------------- | -----: | -----: | -----: | -----: |
| LETTER paper                | 0.0431 | 0.0286 | 0.0672 | 0.0364 |
| Released code + author IDs  | 0.0413 | 0.0268 | 0.0645 | 0.0343 |
| Cornac + Sentence-T5/RQ-VAE | 0.0356 | 0.0241 | 0.0560 | 0.0307 |
| Cornac + author IDs         | 0.0429 | 0.0282 | 0.0670 | 0.0360 |

The Cornac generator with the authors' assignments is within `0.0002` to `0.0004` of every paper metric. The released-code rerun is also below the published row. The lower end-to-end result uses Sentence-T5 content embeddings and locally trained SASRec embeddings because the authors' tokenizer inputs are unavailable.

### Implementation validation

The tokenizer was compared directly with the released implementation using the same item-aligned input matrices. Initial code assignments and collision handling matched exactly, and the first matched optimization step produced identical IDs and losses with a maximum post-step parameter difference of `1.86e-9`.

Run the focused LETTER tests with:

```bash
python -m pytest tests/cornac/models/letter/test_letter.py -q
```

### Reproduction limitation

The released repository provides the final Beauty Semantic-ID table, but not the trained tokenizer checkpoint, content embeddings, or SASRec checkpoint used to produce it. Exact reproduction of the paper's learned Semantic IDs therefore requires those missing artifacts; the Sentence-T5 result above is a runnable replacement rather than an exact reconstruction of the paper's tokenizer inputs.
