# Copyright 2026 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Reproduce LETTER's Beauty generator evaluation with author Semantic IDs.

Pair the authors' numeric-keyed ``Beauty.index.json`` with their released
``Beauty.inter.json``. Alternatively, use an ASIN-keyed Semantic-ID file with
Cornac's Amazon loader. Tokenizer reproduction is out of scope because its
Beauty inputs are not public.
"""

import argparse
import json
import re

import numpy as np

from cornac.data import FeatureModality
from cornac.datasets import amazon_review
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import NDCG, Recall
from cornac.models import LETTER
from cornac.models.letter import LETTER_BEAUTY_CONFIG


def load_semantic_ids(path, item_id_map_path=None):
    """Load four-level Semantic IDs keyed by raw item ID."""
    with open(path) as stream:
        raw_ids = json.load(stream)
    if item_id_map_path is None:
        item_id_map = None
    else:
        with open(item_id_map_path) as stream:
            item_id_map = json.load(stream)

    semantic_ids = {}
    for source_id, tokens in raw_ids.items():
        if len(tokens) != 4:
            raise ValueError(
                f"semantic ID for {source_id!r} has {len(tokens)} levels; expected 4"
            )
        item_id = source_id if item_id_map is None else item_id_map[str(source_id)]
        codes = []
        for token in tokens:
            if isinstance(token, int):
                codes.append(token)
                continue
            match = re.fullmatch(r"<[a-d]_(\d+)>", token)
            if match is None:
                raise ValueError(f"invalid semantic token {token!r}")
            codes.append(int(match.group(1)))
        if item_id in semantic_ids:
            raise ValueError(f"duplicate mapped item ID {item_id!r}")
        semantic_ids[item_id] = codes
    return semantic_ids


def align_semantic_ids(eval_method, semantic_ids):
    """Arrange raw-ID-keyed codes in Cornac's global item-index order."""
    missing = set(eval_method.global_iid_map) - set(semantic_ids)
    if missing:
        raise ValueError(f"semantic-ID file is missing {len(missing)} Beauty items")
    aligned = np.empty((eval_method.total_items, 4), dtype="int64")
    for item_id, item_index in eval_method.global_iid_map.items():
        aligned[item_index] = semantic_ids[item_id]
    return aligned


def load_released_feedback(path):
    """Convert the authors' ordered Beauty sequences to Cornac feedback."""
    with open(path) as stream:
        sequences = json.load(stream)
    return [
        (user_id, str(item_id), 1.0, timestamp)
        for user_id, item_ids in sequences.items()
        for timestamp, item_id in enumerate(item_ids)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("semantic_id_file", help="JSON item-to-Semantic-ID mapping")
    parser.add_argument(
        "--item-id-map",
        help="JSON numeric-ID-to-ASIN map for the authors' Beauty.index.json",
    )
    parser.add_argument(
        "--interaction-file",
        help="authors' Beauty.inter.json; otherwise use Cornac's Amazon loader",
    )
    parser.add_argument(
        "--beams",
        nargs="+",
        type=int,
        default=[20, 50],
        help="beam widths evaluated from the same trained generator",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if any(width <= 0 for width in args.beams):
        parser.error("--beams values must be positive")

    semantic_ids = load_semantic_ids(args.semantic_id_file, args.item_id_map)
    item_ids = list(semantic_ids)
    feedback = (
        amazon_review.load_feedback("beauty")
        if args.interaction_file is None
        else load_released_feedback(args.interaction_file)
    )
    eval_method = NextItemEvaluation.leave_last_out(
        feedback,
        fmt="UIRT",
        mode="last",
        exclude_unknowns=False,
        item_feature=FeatureModality(
            features=np.zeros((len(item_ids), 1), dtype="float32"),
            ids=item_ids,
        ),
        verbose=True,
    )
    aligned_ids = align_semantic_ids(eval_method, semantic_ids)

    config = dict(LETTER_BEAUTY_CONFIG)
    config.update(
        precomputed_semantic_ids=aligned_ids,
        n_beams=args.beams[0],
        device=args.device,
        seed=42,
        verbose=True,
    )
    model = LETTER(name=f"LETTER-b{args.beams[0]}", **config)
    metrics = [Recall(k=5), NDCG(k=5), Recall(k=10), NDCG(k=10)]

    results = {}
    for index, beam_width in enumerate(dict.fromkeys(args.beams)):
        if index:
            model.trainable = False
            model.n_beams = beam_width
            model.name = f"LETTER-b{beam_width}"
        test_result, _ = eval_method.evaluate(
            model, metrics=metrics, user_based=False, show_validation=False
        )
        results[str(beam_width)] = {
            key: float(value)
            for key, value in test_result.metric_avg_results.items()
            if "(s)" not in key
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
