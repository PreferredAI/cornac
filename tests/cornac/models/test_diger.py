# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Small CPU checks for DIGER training and its optional semantic-ID adapter."""

import os
import subprocess
import warnings
import sys
from unittest.mock import patch

import numpy as np
import pytest

from cornac.data import FeatureModality
from cornac.eval_methods import NextItemEvaluation
from cornac.metrics import Recall
from cornac.models import DIGER


def tiny_evaluation(semantic_id_metrics=False):
    return NextItemEvaluation.from_splits(
        train_data=[("u", "tr", "a", 0), ("u", "tr", "b", 1),
                    ("u", "tr", "c", 2)],
        val_data=[("u", "va", "b", 3), ("u", "va", "a", 4),
                  ("u", "va", "c", 5)],
        test_data=[("u", "te", "a", 6), ("u", "te", "b", 7)],
        fmt="USIT", mode="last", semantic_id_metrics=semantic_id_metrics,
        item_feature=FeatureModality(features=np.eye(3), ids=["a", "b", "c"]),
    )


def tiny_model(**kwargs):
    return DIGER(
        tokenizer_options={"hidden_dims": (4,), "latent_dim": 2,
                           "levels": 1, "codebook_size": 4},
        generator_options={"model_dim": 4, "ff_dim": 8, "num_layers": 1,
                           "num_heads": 1, "key_dim": 4, "dropout": 0},
        initializer_options={"epochs": 1, "warmup_epochs": 1, "batch_size": 3},
        n_epochs=1, batch_size=2, eval_batch_size=1, n_beams=3,
        max_len=2, warmup_steps=0, seed=7, initializer_seed=7, **kwargs,
    )


@pytest.fixture(scope="module")
def fitted():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    evaluation = tiny_evaluation()
    model = tiny_model().fit(evaluation.train_set, evaluation.val_set)
    yield model, evaluation
    torch.set_num_threads(old_threads)


def test_optional_dependencies_load_only_when_used():
    script = """
import sys
class BlockBackend:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise ImportError('optional backend blocked')
sys.meta_path.insert(0, BlockBackend())
from cornac.models import DIGER
DIGER()
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


def test_prefixes_and_feature_alignment():
    evaluation = tiny_evaluation()
    model = DIGER(trainable=False).fit(evaluation.train_set, evaluation.val_set)
    assert model._rows(evaluation.train_set) == [([1], 2), ([1, 2], 3)]
    assert model._rows(evaluation.val_set) == [([2, 1], 3)]
    features, ids = model._feature_rows(evaluation.train_set)
    np.testing.assert_array_equal(features, np.eye(3))
    assert ids == ["a", "b", "c"]
    evaluation.train_set.item_feature.ids = ["c", "b", "a"]
    with pytest.raises(ValueError, match="align"):
        model._feature_rows(evaluation.train_set)


def test_catalog_and_paired_checkpoint(fitted):
    model, _ = fitted
    np.testing.assert_array_equal(model.semantic_id_catalog(), model.codes_[1:])
    assert model.semantic_id_catalog().shape == (3, 2)
    assert len(set(map(tuple, model.semantic_id_catalog()))) == 3
    np.testing.assert_array_equal(model.backend_.catalog().cpu(), model.codes_)
    assert model.best_epoch == model.training_summary_["best_epoch"]
    assert all(np.isfinite(row["loss"]) for row in model.training_summary_["history"])


def test_shared_inference_keeps_raw_beam_rows(fitted):
    import torch

    model, _ = fitted
    valid = model.codes_[1].copy()
    raw = np.stack([valid, [99, 99], valid])
    paths = torch.tensor(raw)[None]
    values = torch.tensor([[-1., -2., -3.]])
    with patch.object(model.backend_.generator, "beam_search", return_value=(paths, values)) as beam:
        ranks, scores, semantic_ids = model.rank_with_semantic_ids(0, history_items=[0], k=2)
        assert beam.call_count == 1
        assert beam.call_args.kwargs["beams"] == model.n_beams
        np.testing.assert_array_equal(semantic_ids, raw)
        assert scores[0] == -3  # Preserve the native address-to-item score mapping.
        assert np.isneginf(scores[1:]).all()
        assert ranks[0] == 0


def test_empty_and_unknown_histories(fitted):
    model, _ = fitted
    scores, ids = model._score_with_semantic_ids(0, [])
    assert np.isneginf(scores).all()
    assert ids.shape == (0, 2)
    with pytest.raises(ValueError, match="outside catalog"):
        model.score(0, [model.total_items])


def test_save_load_and_cpu_fallback(fitted, tmp_path):
    import torch

    model, _ = fitted
    before = model.score(0, [0, 1])
    path = model.save(str(tmp_path))
    loaded = DIGER.load(path)
    loaded.device = "cuda"
    with patch.object(torch.cuda, "is_available", return_value=False):
        np.testing.assert_array_equal(loaded.score(0, [0, 1]), before)
    assert loaded.device == "cpu"
    np.testing.assert_array_equal(loaded.codes_, model.codes_)


def test_fit_reproducibility_and_opt_in_evaluation(fitted):
    model, _ = fitted
    evaluation = tiny_evaluation(semantic_id_metrics=True)
    repeated = tiny_model().fit(evaluation.train_set, evaluation.val_set)
    np.testing.assert_array_equal(repeated.codes_, model.codes_)
    np.testing.assert_array_equal(repeated.score(0, [0]), model.score(0, [0]))
    repeated.trainable = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, _ = evaluation.evaluate(repeated, [Recall(k=1)], user_based=False)
    assert all("already fitted" in str(w.message) or
               "valid distinct Semantic IDs" in str(w.message) for w in caught)
    assert {"Recall@1", "SID-Recall@1", "CCE-Recall@1"} <= set(result.metric_avg_results)


def test_decoder_rejects_oversized_beams(fitted):
    import torch

    model, _ = fitted
    codes = torch.tensor(model.codes_[[1]])[None]
    with pytest.raises(ValueError, match="vocabulary"):
        model.backend_.generator.beam_search(codes, torch.ones((1, 1), dtype=torch.bool), beams=5)
