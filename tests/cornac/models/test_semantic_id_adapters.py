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

"""Semantic ID adapters share decoding without changing native item scoring."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from cornac.models import TIGER


def tiny_tiger(model_class=TIGER):
    torch.manual_seed(7)
    model = model_class(
        rqvae_num_levels=2,
        rqvae_codebook_size=2,
        n_epochs=0,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        num_enc_layers=1,
        num_dec_layers=1,
        max_len=2,
        dropout=0,
        n_beams=2,
        device="cpu",
        seed=7,
    )
    model.num_items = 3
    model.device_ = "cpu"
    model._build_semantic_ids(np.array([[0, 0], [0, 0], [1, 1]]))
    model._fit_seq2seq(torch, None)
    return model


def native_tiger_prediction(model, history, budget):
    """Direct decoder invocation with master's padding and encoder layout."""
    hist = list(history)[-model.max_len :]
    hist = [model.pad_idx] * (model.max_len - len(hist)) + hist
    tokens = torch.tensor(model.enc_token_table[hist].reshape(1, -1))
    mask = (tokens != 0).float()
    model.model.eval()
    with torch.no_grad():
        if model.scoring == "beam":
            return model.model.generate_beam(
                tokens, mask, budget, model.prefix_children
            )
        return model.model.score_all_items(
            tokens, mask, torch.tensor(model.sid_table), model.scoring_batch_size
        )


def tiny_letter():
    from cornac.data import FeatureModality
    from cornac.eval_methods import NextItemEvaluation
    from cornac.models import LETTER

    evaluation = NextItemEvaluation.from_splits(
        [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)],
        [("u", "te", "a", 3), ("u", "te", "b", 4)],
        fmt="USIT",
        item_feature=FeatureModality(features=np.eye(3), ids=["a", "b", "c"]),
    )
    model = LETTER(
        precomputed_semantic_ids=np.array([[0, 0], [0, 0], [1, 1]]),
        rqvae_num_levels=2,
        rqvae_codebook_size=2,
        n_epochs=1,
        d_model=8,
        d_ff=16,
        num_heads=2,
        d_kv=4,
        letter_base_vocab_size=8,
        num_enc_layers=1,
        num_dec_layers=1,
        max_len=2,
        dropout=0,
        n_beams=2,
        device="cpu",
        seed=7,
    )
    model.fit(evaluation.train_set)
    return model, evaluation


def tiny_rpg(monkeypatch):
    from cornac.data import FeatureModality
    from cornac.eval_methods import NextItemEvaluation
    from cornac.models import RPG

    evaluation = NextItemEvaluation.from_splits(
        [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)],
        [("u", "te", "a", 3), ("u", "te", "b", 4)],
        fmt="USIT",
        item_feature=FeatureModality(features=np.eye(3), ids=["a", "b", "c"]),
    )
    model = RPG(
        n_codebook=2,
        codebook_size=2,
        n_epochs=1,
        d_model=8,
        n_layer=1,
        n_head=2,
        n_inner=16,
        max_len=2,
        resid_dropout=0,
        embd_dropout=0,
        attn_dropout=0,
        n_beams=2,
        graph_edges=2,
        propagation_steps=1,
        device="cpu",
        seed=7,
    )
    # faiss OPQ uses 8-bit PQ and needs >=256 training items; freeze only
    # tokenization to keep this real training/graph-building CPU smoke tiny.
    with monkeypatch.context() as patch:
        patch.setattr(
            RPG,
            "_opq_tokenize",
            lambda self, feats, train_mask: np.array([[0, 0], [0, 0], [1, 1]]),
        )
        model.fit(evaluation.train_set)
    return model, evaluation


class ConstantCPD:
    n_digit = 2
    codebook_size = 2

    def decode_logits(self, memory, padding, partial):
        return torch.tensor([[[4.0, 0.0], [3.0, 0.0]]]).expand(len(partial), -1, -1)


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
def test_cpd_raw_single_batch_default_parity(diagnostics, constrained):
    from cornac.models.diffgrm.diffgrm import cpd_decode, cpd_decode_batch

    memory, padding = torch.zeros(2, 1, 4), torch.zeros(2, 1, dtype=torch.bool)
    kwargs = dict(
        beam_size=4,
        catalog_codes=np.array([[0, 0], [1, 1]]),
        constrained=constrained,
        return_diagnostics=diagnostics,
    )
    default = cpd_decode_batch(ConstantCPD(), memory, padding, **kwargs)
    extra = cpd_decode_batch(ConstantCPD(), memory, padding, return_raw=True, **kwargs)
    assert extra[:-1] == default
    raw = extra[-1]
    assert len(raw) == 2
    for entry in raw:
        assert (
            len(entry["address_sids"])
            == len(entry["scores"])
            == len(entry["catalog_validity"])
            == 4
        )
        # Parallel assignment orders produce repeated complete paths.
        assert len(set(entry["address_sids"])) < len(entry["address_sids"])
        assert entry["catalog_validity"] == [
            sid in {(0, 0), (1, 1)} and np.isfinite(score)
            for sid, score in zip(entry["address_sids"], entry["scores"])
        ]
    single = cpd_decode(
        ConstantCPD(), memory[:1], padding[:1], return_raw=True, **kwargs
    )
    assert single == tuple(part[0] for part in extra)
    legacy_single = cpd_decode(ConstantCPD(), memory[:1], padding[:1], **kwargs)
    assert legacy_single == tuple(part[0] for part in default)
    with pytest.raises(ValueError, match="one encoded history"):
        cpd_decode(ConstantCPD(), memory, padding, **kwargs)


def test_cpd_raw_marks_nonfinite_and_invalid_slots():
    from cornac.models.diffgrm.diffgrm import cpd_decode

    memory, padding = torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.bool)
    codes, scores, raw = cpd_decode(
        ConstantCPD(),
        memory,
        padding,
        beam_size=16,
        catalog_codes=np.array([[1, 1]]),
        return_raw=True,
    )
    assert len(raw["address_sids"]) > len(codes)
    assert any(not valid for valid in raw["catalog_validity"])
    for sid, score, valid in zip(
        raw["address_sids"], raw["scores"], raw["catalog_validity"]
    ):
        assert valid == (sid == (1, 1) and -1 not in sid and np.isfinite(score))
    with pytest.raises(ValueError, match="permutation"):
        cpd_decode(ConstantCPD(), memory, padding, beam_size=2, digit_order=[0, 0])


def tiny_diffgrm():
    from cornac.eval_methods import NextItemEvaluation
    from cornac.models import DiffGRM

    evaluation = NextItemEvaluation.from_splits(
        [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)],
        [("u", "te", "a", 3), ("u", "te", "b", 4)],
        fmt="USIT",
    )
    model = DiffGRM(
        item_sids=np.array([[0, 0], [0, 0], [1, 1]]),
        n_digit=2,
        codebook_size=2,
        d_model=8,
        encoder_n_layer=1,
        decoder_n_layer=1,
        n_head=2,
        n_inner=16,
        dropout=0,
        max_len=2,
        min_history=1,
        n_epochs=1,
        n_views=1,
        beam_size=4,
        lr_schedule="constant",
        device="cpu",
        seed=7,
    )
    model.fit(evaluation.train_set)
    return model, evaluation


@pytest.mark.parametrize("scoring", ["beam", "exact"])
def test_tiger_native_item_parity_and_full_ids(scoring, tmp_path):
    model = tiny_tiger()
    model.scoring = scoring
    native = native_tiger_prediction(model, [0], model.n_beams)
    if scoring == "beam":
        sids, scores = native
        expected = np.full(3, -1e10)
        for sid, score in zip(sids, scores):
            expected[model.sid_to_item[sid]] = score
    else:
        expected = native
        sids = model._semantic_ids_from_scores(expected)
    combined = model.rank_with_semantic_ids(0, np.arange(3), history_items=[0])
    np.testing.assert_allclose(combined[1], expected)
    np.testing.assert_array_equal(combined[2], sids)
    np.testing.assert_allclose(model.score(0, [0]), expected)
    assert not np.array_equal(
        model.semantic_id_catalog()[0], model.semantic_id_catalog()[1]
    )
    restored = model.load(model.save(str(tmp_path)))
    np.testing.assert_array_equal(
        restored.rank_with_semantic_ids(0, history_items=[0])[2], combined[2]
    )
    assert model.rank_with_semantic_ids(0, history_items=[])[2].shape == (0, 3)


@pytest.mark.parametrize("scoring", ["beam", "exact"])
def test_letter_native_item_parity_and_collision_catalog(scoring):
    model, evaluation = tiny_letter()
    model.scoring = scoring
    inputs, mask = model._encoder_batch(torch, [[0]])
    model.model.eval()
    with torch.no_grad():
        if scoring == "beam":
            sids, scores = model.model.generate_beam(
                inputs, mask, model.n_beams, model.prefix_children
            )
            expected = np.full(3, -1e10)
            for sid, score in zip(sids, scores):
                expected[model.sid_to_item[sid]] = score
        else:
            expected = model.model.score_all_items(
                inputs, mask, torch.tensor(model.sid_table), model.scoring_batch_size
            )
            sids = model._semantic_ids_from_scores(expected)
    combined = model.rank_with_semantic_ids(0, np.arange(3), history_items=[0])
    np.testing.assert_allclose(combined[1], expected)
    np.testing.assert_array_equal(combined[2], sids)
    np.testing.assert_allclose(model.score(0, [0]), expected)
    np.testing.assert_array_equal(
        model.semantic_id_catalog()[0], model.semantic_id_catalog()[1]
    )
    assert model.rank_with_semantic_ids(0, history_items=[])[2].shape == (0, 2)


@pytest.mark.parametrize("kind", ["tiger", "letter"])
def test_beam_single_inference_keeps_raw_rows_and_configured_budget(kind, monkeypatch):
    model = tiny_tiger() if kind == "tiger" else tiny_letter()[0]
    sid = tuple(model.sid_table[0])
    invalid = (9,) * len(sid)
    output = [sid, invalid, sid]
    calls = []

    def decode(tokens, mask, budget, children):
        calls.append(budget)
        return output, [-1.0, -2.0, -3.0]

    monkeypatch.setattr(model.model, "generate_beam", decode)
    combined = model.rank_with_semantic_ids(0, [0, 1, 2], k=1, history_items=[0])
    assert calls == [model.n_beams]
    np.testing.assert_array_equal(combined[2], output)


@pytest.mark.parametrize("scoring", ["graph", "exact"])
def test_rpg_native_parity_and_single_rng_advance(monkeypatch, scoring, tmp_path):
    from cornac.models.rpg.rpg import graph_decode

    model, evaluation = tiny_rpg(monkeypatch)
    model.scoring = scoring
    state = model.graph_rng.get_state()
    model.model.eval()
    with torch.no_grad():
        inputs = torch.tensor([[model.pad_idx, 0]])
        logits = model.model.next_item_logits(
            inputs, (inputs != model.pad_idx).float()
        )[0]
    if scoring == "graph":
        candidates, scores = graph_decode(
            logits.reshape(-1).numpy(),
            model.adjacency,
            model.item_tokens,
            model.n_beams,
            model.propagation_steps,
            model.graph_rng,
        )
        after_native = model.graph_rng.get_state()
        expected = np.full(3, -1e10)
        expected[candidates] = scores
        expected_sids = model.sid_table[candidates]
        model.graph_rng.set_state(state)
    else:
        expected = model._exact_item_scores(logits)[0].numpy().astype(float)
        expected_sids = model._semantic_ids_from_scores(expected)
    combined = model.rank_with_semantic_ids(0, np.arange(3), history_items=[0])
    np.testing.assert_allclose(combined[1], expected)
    np.testing.assert_array_equal(combined[2], expected_sids)
    if scoring == "graph":
        assert all(
            np.array_equal(a, b)
            for a, b in zip(after_native, model.graph_rng.get_state())
        )
    model.graph_rng.set_state(state)
    ordinary = model.rank(0, np.arange(3), history_items=[0])
    np.testing.assert_array_equal(combined[0], ordinary[0])
    np.testing.assert_allclose(combined[1], ordinary[1])
    restored = model.load(model.save(str(tmp_path)))
    np.testing.assert_array_equal(
        restored.rank_with_semantic_ids(0, history_items=[0])[2],
        model.rank_with_semantic_ids(0, history_items=[0])[2],
    )
    assert model.rank_with_semantic_ids(0, history_items=[])[2].shape == (0, 2)


@pytest.mark.parametrize("scoring", ["released", "paper", "catalog", "fixed"])
@pytest.mark.parametrize("policy", ["all", "first", "last"])
def test_diffgrm_native_parity_and_raw_sids(scoring, policy):
    from cornac.models.diffgrm.diffgrm import cpd_decode

    model, evaluation = tiny_diffgrm()
    model.scoring = scoring
    model.collision_policy = policy
    model.model.eval()
    with torch.no_grad():
        inputs = torch.tensor([[0, model.pad_idx]])
        memory, padding = model.model.encode_history(inputs, torch.tensor([[1.0, 0.0]]))
        codes, scores, diagnostics, raw = cpd_decode(
            model.model,
            memory,
            padding,
            beam_size=model.beam_size,
            catalog_codes=model.sid_table,
            valid_code_set=model.sid_to_items_,
            constrained=scoring == "catalog",
            digit_order=model.fixed_decode_order_ if scoring == "fixed" else None,
            greedy_final=scoring == "released",
            return_diagnostics=True,
            return_raw=True,
        )
    expected = np.full(3, -1e10)
    for sid, score in zip(codes, scores):
        members = model.sid_to_items_[sid]
        members = (
            members[:1]
            if policy == "first"
            else members[-1:] if policy == "last" else members
        )
        expected[members] = score
    combined = model.rank_with_semantic_ids(0, np.arange(3), history_items=[0])
    np.testing.assert_allclose(combined[1], expected)
    np.testing.assert_array_equal(combined[2], raw["address_sids"])
    np.testing.assert_allclose(model.score(0, [0]), expected)
    assert model.rank_with_semantic_ids(0, history_items=[])[2].shape == (0, 2)


def test_diffgrm_single_decode_resolution_independence_and_budget(monkeypatch):
    import cornac.models.diffgrm.diffgrm as backend
    from cornac.metrics import Recall

    model, evaluation = tiny_diffgrm()
    model.trainable = False
    original = backend.cpd_decode
    calls = []

    def decode(*args, **kwargs):
        calls.append(kwargs["beam_size"])
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "cpd_decode", decode)
    rows = []
    for policy in ("all", "first", "last"):
        model.collision_policy = policy
        result = evaluation.eval(
            model,
            evaluation.train_set,
            evaluation.test_set,
            True,
            [Recall(1)],
            semantic_id_metrics=True,
        )
        rows.append(
            [
                result.metric_avg_results[name]
                for name in ("SID-Recall@1", "CCE-Recall@1")
            ]
        )
    np.testing.assert_allclose(rows, [rows[0]] * 3)
    assert calls == [model.beam_size] * 3
