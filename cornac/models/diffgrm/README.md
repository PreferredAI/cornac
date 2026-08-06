# DiffGRM

Cornac implementation of **DiffGRM: Diffusion-based Generative Recommendation Model** (WWW 2026, [paper](https://arxiv.org/abs/2510.21805)). DiffGRM is a generative next-item recommender that represents each catalog item as a short Semantic ID and predicts all positions through a masked-diffusion process rather than autoregressive left-to-right generation.

DiffGRM combines three mechanisms:

1. **Parallel Semantic Encoding (PSE)** whitens item-content embeddings with PCA and uses OPQ/PQ to create four independent 8-bit Semantic-ID digits.
2. **On-policy Coherent Noising (OCN)** probes a fully masked target, ranks its digits by confidence, and constructs nested masked views that focus learning on the most uncertain digits.
3. **Confidence-guided Parallel Denoising (CPD)** lets every unfilled digit/code pair compete within a global beam, so the decoding order is selected from the model's confidence rather than fixed in advance.

The implementation was independently written for Cornac from the published architecture and equations. The [released research repository](https://github.com/liuzhao09/DiffGRM) was audited at commit `ad7b971c7e525e9fea6fb8e362a5c49dccb2473c` to validate public behavior and resolve ambiguities between the paper and release (reproduce the results from the official repo and compare against the paper). That commit did not include a root repository license, so its source code was not copied into Cornac.

## Requirements

Install the optional DiffGRM dependencies listed in `requirements.txt`:

```bash
pip install -r cornac/models/diffgrm/requirements.txt
```

DiffGRM requires `torch`, `faiss-cpu`, and `scikit-learn`.

## Usage

DiffGRM consumes precomputed item-content embeddings through Cornac's `FeatureModality`:

```python
from cornac.data import FeatureModality
from cornac.eval_methods import NextItemEvaluation
from cornac.models.diffgrm import DiffGRM, DIFFGRM_SPORTS_CONFIG

item_feature = FeatureModality(features=item_embeddings, ids=item_ids)
eval_method = NextItemEvaluation.from_splits(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    fmt="USIT",
    item_feature=item_feature,
)

model = DiffGRM(**{**DIFFGRM_SPORTS_CONFIG, "seed": 2026})
```

`item_embeddings` must cover every mapped item, including validation and test items, because the fitted transform is applied to the full recommendation catalog. PCA and OPQ/PQ are fitted only on items exposed by the released training-row construction; held-out interactions do not enter that fit. Sentence-T5 encoding is an offline preprocessing step and is never downloaded implicitly by the model.

For artifact-controlled experiments, pass an integer array with shape `(n_items, n_digit)` through `item_sids`. This bypasses PSE and is the recommended way to compare model behavior while keeping a published Semantic-ID artifact fixed.

The released sliding augmenter creates prefix targets from `min_history` through `max_len` and stops after the first `max_len` target. It does not roll a length-`max_len` window over later targets in a longer sequence. Cornac follows this released behavior, and PSE fitting uses the union of items exposed by those exact training rows.

## Training and inference controls

### Masking and loss

`masking_strategy="guided"` enables OCN. The alternatives are designed as explicit controls: `"random"` independently masks digits for the no-OCN comparison, `"coherent"` creates nested random-order masks without on-policy selection, and `"fixed"` always uses a fixed digit order. The released no-OCN recipe uses four independent views with `random_mask_prob=0.5`.

The paper averages the masked loss within each view and then across views, exposed as `view_loss_reduction="view_mean"`. The released implementation pools all masked tokens across views instead, exposed as `view_loss_reduction="token_mean"`.

### Decoding

The `scoring` option separates the paper algorithm, released behavior, and controlled ablations:

- `scoring="paper"` follows Equations 8--10 and performs global beam selection at every denoising step.
- `scoring="released"` reproduces the released CPD behavior, including greedy completion of the final digit, followed by complete-catalog filtering.
- `scoring="catalog"` applies paper CPD while constraining every partial assignment to catalog-compatible prefixes.
- `scoring="fixed"` uses a seeded fixed digit permutation and serves as the no-CPD control.

Paper-style configurations use validation beam 32 through `val_beam_size` and the dataset-specific `beam_size` for test scoring. Validation decoding is batched according to `val_batch_size`.

### Collisions and model selection

PSE can map multiple items to the same Semantic ID. The default `collision_policy="all"` assigns a decoded path score to every item with that ID. `"last"` reproduces the released reverse-map behavior, where the last catalog item sharing an ID overwrites earlier items, while `"first"` retains only the lowest-index item.

Checkpoint selection defaults to `model_selection="best"` and maximizes the SID-level validation objective `0.8 * NDCG@k + 0.2 * Recall@k`. SID-level selection ranks the target Semantic ID before expanding collisions to items; it does not regenerate IDs or use test data. Use `model_selection="last"` only when the final epoch is intentionally required or no validation split is available.

After fitting, tokenizer and collision diagnostics are available through `tokenizer_time_`, `sid_hash_`, `sid_digit_utilization_`, `sid_digit_entropy_`, `sid_collision_count_`, `sid_collision_group_count_`, and `sid_max_collision_size_`. Training records `training_time_` and `loss_history_`; scoring records `last_decode_time_` and `last_decode_diagnostics_`.

## Paper and released-code differences

The paper and released repository differ in several consequential details. Cornac keeps these choices visible rather than silently blending them into one recipe.

| Behavior                   | Paper                                    | Released repository                      | Cornac control                    |
| -------------------------- | ---------------------------------------- | ---------------------------------------- | --------------------------------- |
| Multi-view loss            | Mean within each view, then across views | Pool all masked tokens across views      | `view_loss_reduction`             |
| Final CPD step             | Global beam selection through completion | Greedy final code per active branch      | `scoring="paper"` or `"released"` |
| Test beam                  | Sports/Beauty/Toys: 128/256/128          | Shared default 256                       | Dataset-specific `beam_size`      |
| Maximum epochs             | 100                                      | Commands inherit 200                     | `n_epochs`                        |
| Beauty label smoothing     | 0.1                                      | Command passes 0.2                       | `label_smoothing`                 |
| Long-sequence augmentation | Described as all contiguous subsequences | Prefix targets only through `max_len=50` | Cornac follows the released rows  |
| Semantic-ID collisions     | Item resolution is underspecified        | Last item overwrites earlier items       | `collision_policy`                |

The backbone follows the audited released architecture: pre-normalized bias-free attention, normal initialization with standard deviation 0.02, a final normalization shared by the encoder and decoder, zeroed padded encoder states, and decoder cross-attention over those states without a padding mask. The Sports backbone contains 5,601,280 parameters. Cornac omits three behaviorally unused BOS/EOS/PAD embedding rows, accounting for the released model's additional 768 parameters.

PSE uses the released 32-thread FAISS setting and FAISS's default clustering seeds; the model seed is not substituted for FAISS's defaults. PCA and FAISS artifacts are dependency-version sensitive, so algorithmically equivalent environments may still generate different code tables.

## Sports reproducibility study

### Scope and protocol

The completed controlled study focuses on the Amazon Reviews 2014 Sports and Outdoors 5-core dataset used by the paper. The processed split contains 35,598 users, 18,357 items, 152,346 released-style training examples, and 35,598 validation and test cases each. Every controlled Cornac run uses the same split and frozen released Semantic-ID table, so differences after tokenization come from model initialization, minibatch and masking order, dropout, checkpoint selection, and decoding rather than regenerated item codes.

The Cornac comparison uses declared seeds 2024, 2025, and 2026 on a single NVIDIA A40 per run. It keeps the released pooled-token loss, released CPD, validation beam 32, paper test beam 128, SID-level validation objective, and `collision_policy="all"`. Held-out test metrics are reported only after validation-based checkpoint selection and are not used to choose a seed or checkpoint.

Metrics in the primary reproduction table are SID-level, matching the released evaluator. Item-expanded metrics are reported separately because collisions make SID retrieval and exact item recommendation different objectives.

### Reproduction path

The evaluation separated training changes from checkpoint-only rescoring:

1. Run the released repository with its documented Sports command.
2. Rescore the same checkpoint with the paper beam to isolate beam width.
3. Retrain with the paper's per-view loss while holding the remaining released recipe fixed.
4. Regenerate embeddings and Semantic IDs without the released-only `Features` metadata field to test the paper metadata interpretation.
5. Run the Cornac adapter with the frozen released IDs, correct SID-level checkpoint selection, and the audited release-fidelity backbone.

| Source              | Recipe                                            |         Seed | Selected epoch | Test beam |          Recall@5 |            NDCG@5 |         Recall@10 |           NDCG@10 |
| ------------------- | ------------------------------------------------- | -----------: | -------------: | --------: | ----------------: | ----------------: | ----------------: | ----------------: |
| Paper               | Reported DiffGRM                                  | Not reported |   Not reported |       128 |             .0363 |             .0245 |             .0550 |             .0305 |
| Released repository | As-released training and decoding                 |         2024 |             54 |       256 |             .0329 |             .0223 |             .0502 |             .0279 |
| Released repository | Same checkpoint, paper beam                       |         2024 |             54 |       128 |             .0329 |             .0223 |             .0500 |             .0278 |
| Released repository | Paper per-view loss; released decoding            |         2024 |             44 |       128 |             .0337 |             .0225 |             .0516 |             .0283 |
| Released repository | Paper metadata fields; released pooled-token loss |         2024 |             28 |       128 |             .0366 |             .0242 |             .0558 |             .0303 |
| Cornac              | Release-fidelity SID-selected                     |         2024 |             49 |       128 |             .0324 |             .0220 |             .0501 |             .0277 |
| Cornac              | Release-fidelity SID-selected                     |         2025 |             26 |       128 |             .0360 |             .0244 |             .0545 |             .0304 |
| Cornac              | Release-fidelity SID-selected                     |         2026 |             25 |       128 |             .0325 |             .0218 |             .0518 |             .0280 |
| Cornac              | Release-fidelity mean $\pm$ sample SD             |   2024--2026 |             -- |       128 | .0337 $\pm$ .0020 | .0227 $\pm$ .0014 | .0521 $\pm$ .0022 | .0287 $\pm$ .0015 |

### What explains the paper gap

The as-released run is below the paper on all four Sports metrics; its Recall@10 and NDCG@10 gaps are `-8.73%` and `-8.65%`. Changing only the beam from 256 to the paper's 128 does not close the gap. Paper-style per-view loss improves the selected checkpoint, but Recall@10 and NDCG@10 remain `-6.12%` and `-7.36%` below the paper.

The largest observed change comes from metadata preprocessing. The released data path includes a `Features` field that is absent from the paper's stated text fields. Removing that field before regenerating embeddings and Semantic IDs puts the single controlled run within `1.5%` of all four paper metrics: Recall@5 and Recall@10 are `+0.84%` and `+1.49%`, while NDCG@5 and NDCG@10 are `-1.42%` and `-0.49%`. This isolates metadata construction as the main observed source of the released-to-paper gap, but the conclusion is based on one seed.

### SID selection and collision effects

The frozen Sports catalog contains 18,357 items but only 15,448 unique Semantic IDs. There are 2,909 item-to-ID collisions across 1,394 collision groups, with a maximum group size of 47. The released last-item reverse map discards all but one item from each collided ID, whereas the controlled Cornac runs retain every collided item through `collision_policy="all"`.

Early Cornac integration runs selected checkpoints after expanding decoded IDs to items. That item-level criterion is not equivalent to the released SID-level validation objective when IDs collide. Retraining the same three seeds with SID-level selection raises the mean of every SID test metric by `2.77%`--`3.96%` and reduces their observed sample standard deviations by `66.66%`--`74.43%`. The corresponding item-expanded means decrease by `3.87%`--`4.90%`, demonstrating a real objective tradeoff rather than a universally better checkpoint.

This correction also illustrates why the README reports both endpoints explicitly. SID-level metrics measure recovery of the target code and are directly comparable with the released evaluator; item-level metrics measure exact catalog recommendation after resolving collisions.

### Release-fidelity comparison

The final architecture was chosen through a controlled comparison against the initial corrected Cornac backbone. Both variants used the same split, frozen IDs, seeds, optimization settings, released scoring, SID-level validation selector, and collision policy. The release-fidelity variant changed the attention and normalization layout, initialization, and cross-attention padding behavior to match the audited release. The frozen IDs intentionally bypassed PSE, while released FAISS behavior and last-item collision resolution were validated separately.

| Metric                   | Corrected Cornac baseline | Release-fidelity implementation | Relative change |
| ------------------------ | ------------------------: | ------------------------------: | --------------: |
| SID validation objective |                   .033613 |                         .033842 |          +0.68% |
| SID Recall@5             |                   .032745 |                         .033654 |          +2.77% |
| SID NDCG@5               |                   .021557 |                         .022734 |          +5.46% |
| SID Recall@10            |                   .049881 |                         .052147 |          +4.54% |
| SID NDCG@10              |                   .027062 |                         .028676 |          +5.96% |
| Item Recall@5            |                   .015844 |                         .015937 |          +0.59% |
| Item NDCG@5              |                   .010761 |                         .011201 |          +4.09% |
| Item Recall@10           |                   .024543 |                         .024571 |          +0.11% |
| Item NDCG@10             |                   .013538 |                         .013955 |          +3.07% |
| Valid-path fraction      |                    95.87% |                          95.74% |          -0.13% |
| Mean stop epoch          |                      69.0 |                            48.3 |         -29.95% |
| Mean training time       |                 107.8 min |                        61.4 min |         -42.99% |
| Mean test decode time    |                   191.5 s |                         154.7 s |         -19.21% |

The release-fidelity implementation improves the mean validation objective and all eight SID/item ranking metrics while preserving the valid-path rate. It also stops earlier and completes sooner. The runtime difference is partly caused by earlier early stopping, so it is not evidence of a pure per-epoch optimization. The release-fidelity sample standard deviation is higher for all four SID test metrics, so the higher means should not be described as reduced seed sensitivity.

Relative to the paper, the retained three-seed mean is close but does not match the reported Sports result:

| Metric    | Paper | Cornac mean | Mean gap | Seed 2025 | Seed gap |
| --------- | ----: | ----------: | -------: | --------: | -------: |
| Recall@5  | .0363 |     .033654 |   -7.29% |   .036013 |   -0.79% |
| NDCG@5    | .0245 |     .022734 |   -7.21% |   .024388 |   -0.46% |
| Recall@10 | .0550 |     .052147 |   -5.19% |   .054497 |   -0.91% |
| NDCG@10   | .0305 |     .028676 |   -5.98% |   .030353 |   -0.48% |

Seed 2025 also has the highest validation score among the three declared seeds and is within 1% of every paper metric. The aggregate remains the primary reproduction result because the paper does not report its seed or variance, and test performance was not used to select among the Cornac seeds.

### PSE artifact fidelity

The PSE audit exactly reproduced all 18,357 released Sports Semantic-ID rows when it began from the released cached PCA matrix and ran the OPQ/PQ stage with FAISS 1.11.0 and scikit-learn 1.7.0. This establishes exact agreement for the changed Cornac PSE stage under the released input and dependency environment.

Full regeneration from raw content embeddings through PCA did not reproduce the cached PCA-derived IDs, even with those dependency versions. The released determinism check also starts from the cached PCA matrix, so this is an unresolved artifact-provenance boundary rather than an end-to-end PSE reproduction. Newer supported versions of FAISS and scikit-learn can produce different code tables while following the same algorithm; use frozen `item_sids` whenever exact artifact identity matters.

### Interpretation and remaining limits

The completed Sports study supports the following conclusions:

- The Cornac implementation is behaviorally close to the published Sports result, but its three-seed mean remains `5.19%`--`7.29%` lower across the four reported metrics.
- A validation-selected Cornac seed is within 1% of all four paper metrics, but that individual result does not replace the multi-seed aggregate.
- Metadata preprocessing is the main observed explanation for the released-to-paper gap in the controlled single-seed diagnostics.
- SID-level and exact-item evaluation answer different questions when the tokenizer has collisions; neither should be silently substituted for the other.
- The retained backbone is closer to the released architecture and improves the controlled validation objective and mean ranking metrics, but three seeds are insufficient for a precise variance claim.
- Exact PSE reproduction currently requires the released cached PCA matrix or frozen Semantic IDs; raw-to-PCA provenance remains unresolved.

The study does not yet constitute a full reproduction of the entire paper. The released repository has only a one-seed matched reference, the central `random` and `fixed` ablations have not been run end to end in Cornac, and Beauty and Toys have not received the same three-seed evaluation.

## Paper-reported Amazon-2014 results

The following values are references from the paper, not Cornac reproduction claims.

| Dataset | Model   | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| ------- | ------- | -------: | -----: | --------: | ------: |
| Sports  | RPG     |    .0314 |  .0216 |     .0463 |   .0263 |
| Sports  | DiffGRM |    .0363 |  .0245 |     .0550 |   .0305 |
| Beauty  | RPG     |    .0550 |  .0381 |     .0809 |   .0464 |
| Beauty  | DiffGRM |    .0603 |  .0414 |     .0876 |   .0502 |
| Toys    | RPG     |    .0592 |  .0401 |     .0869 |   .0490 |
| Toys    | DiffGRM |    .0618 |  .0455 |     .0834 |   .0524 |

The exported paper configurations use four 256-way digits, one encoder layer, four decoder layers, 100 epochs, and dataset-specific learning rates, label smoothing, model dimensions, and beam widths.

## References

- Zhao Liu, Yichen Zhu, Yiqing Yang, Guoping Tang, Rui Huang, Qiang Luo, Xiao Lv, Ruiming Tang, Kun Gai, and Guorui Zhou. [DiffGRM: Diffusion-based Generative Recommendation Model](https://arxiv.org/abs/2510.21805). WWW 2026.
- [Released DiffGRM repository](https://github.com/liuzhao09/DiffGRM), behavior audited at commit `ad7b971c7e525e9fea6fb8e362a5c49dccb2473c`.
