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

"""Raw SID adapters retain native scoring and decoder slot semantics."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cornac.models import TIGER
from cornac.utils.semantic_id import prediction_from_catalog_scores


def tiny_tiger(model_class=TIGER):
    torch.manual_seed(7)
    model = model_class(
        rqvae_num_levels=2, rqvae_codebook_size=2, n_epochs=0,
        d_model=8, d_ff=16, num_heads=2, d_kv=4,
        num_enc_layers=1, num_dec_layers=1, max_len=2,
        dropout=0, n_beams=2, device="cpu", seed=7,
    )
    model.num_items = 3
    model.device_ = "cpu"
    model._build_semantic_ids(np.array([[0, 0], [0, 0], [1, 1]]))
    model._fit_seq2seq(torch, None)
    return model


def native_tiger_prediction(model, history, budget):
    """Direct decoder invocation with master's padding and encoder layout."""
    hist = list(history)[-model.max_len:]
    hist = [model.pad_idx] * (model.max_len - len(hist)) + hist
    tokens = torch.tensor(model.enc_token_table[hist].reshape(1, -1))
    mask = (tokens != 0).float()
    model.model.eval()
    with torch.no_grad():
        if model.scoring == "beam":
            return model.model.generate_beam(tokens, mask, budget, model.prefix_children)
        return model.model.score_all_items(
            tokens, mask, torch.tensor(model.sid_table), model.scoring_batch_size
        )


def test_tiger_catalog_and_raw_slots(monkeypatch):
    model = tiny_tiger()
    cat = model.semantic_id_catalog()
    assert cat.tokenizer_sid_for_item(0) == cat.tokenizer_sid_for_item(1)
    assert cat.address_sid_for_item(0) != cat.address_sid_for_item(1)
    a = cat.address_sid_for_item(0)
    outputs = [a, (9, 9, 9), a]
    seen = []

    def decode(tokens, mask, budget, children):
        seen.append(budget)
        return outputs, [-1., -2., -3.]

    monkeypatch.setattr(model.model, "generate_beam", decode)
    pred = model.predict_semantic_ids(np.array([0]), n_candidates=4)
    assert seen == [4]
    assert model.n_beams == 2
    assert pred.address_sids.tolist() == [list(row) for row in outputs]
    assert pred.catalog_validity.tolist() == [True, False, True]
    assert len(model.predict_semantic_ids([]).scores) == 0
    np.testing.assert_equal(model.score(0, []), np.ones(3))


@pytest.mark.parametrize("scoring", ["beam", "exact"])
def test_tiger_native_parity_and_persistence(scoring, tmp_path):
    model = tiny_tiger()
    model.scoring = scoring
    native = native_tiger_prediction(model, [0], model.n_beams)
    if scoring == "beam":
        addresses, logps = native
        expected = np.full(3, -1e10)
        for sid, score in zip(addresses, logps):
            expected[model.sid_to_item[sid]] = score
        raw = model.predict_semantic_ids([0])
        np.testing.assert_array_equal(raw.address_sids, addresses)
        np.testing.assert_allclose(raw.scores, logps)
        enlarged = model.predict_semantic_ids([0], n_candidates=3)
        assert len(enlarged.scores) == 3
    else:
        expected = native
        oracle = prediction_from_catalog_scores(model.semantic_id_catalog(), expected)
        raw = model.predict_semantic_ids([0])
        np.testing.assert_array_equal(raw.address_sids, oracle.address_sids)
        np.testing.assert_allclose(raw.scores, oracle.scores)
    np.testing.assert_allclose(model.score(0, [0]), expected)
    restored = model.load(model.save(str(tmp_path)))
    assert restored.semantic_id_catalog().stable_hash == model.semantic_id_catalog().stable_hash
    np.testing.assert_allclose(restored.score(0, [0]), expected)
    np.testing.assert_array_equal(restored.predict_semantic_ids([0]).address_sids, raw.address_sids)


def test_tiger_cpu_fit_and_experiment(tmp_path):
    from cornac.data import FeatureModality
    from cornac.eval_methods import SemanticIDEvaluation
    from cornac.metrics import Recall

    evaluation = SemanticIDEvaluation.from_splits(
        [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)],
        [("u", "te", "a", 3), ("u", "te", "b", 4)], fmt="USIT",
        item_feature=FeatureModality(features=np.eye(3), ids=["a", "b", "c"]),
    )
    model = TIGER(
        rqvae_num_levels=2, rqvae_codebook_size=2, rqvae_n_epochs=1,
        rqvae_hidden_dims=(4,), rqvae_latent_dim=2, n_epochs=1,
        d_model=8, d_ff=16, num_heads=2, d_kv=4,
        num_enc_layers=1, num_dec_layers=1, max_len=2,
        dropout=0, n_beams=2, device="cpu", seed=7,
    )
    result, _ = evaluation.evaluate(model, [Recall(2)], user_based=False)
    assert {"SID/Recall@2", "CCE/Recall@2", "Item[popularity]/Recall@2"} <= set(result.metric_avg_results)
    restored = model.load(model.save(str(tmp_path)))
    np.testing.assert_array_equal(model.predict_semantic_ids([0]).address_sids,
                                  restored.predict_semantic_ids([0]).address_sids)


def tiny_letter():
    from cornac.data import FeatureModality
    from cornac.eval_methods import SemanticIDEvaluation
    from cornac.models import LETTER

    evaluation = SemanticIDEvaluation.from_splits(
        [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)],
        [("u", "te", "a", 3), ("u", "te", "b", 4)], fmt="USIT",
        item_feature=FeatureModality(features=np.eye(3), ids=["a", "b", "c"]),
    )
    model = LETTER(
        precomputed_semantic_ids=np.array([[0, 0], [0, 0], [1, 1]]),
        rqvae_num_levels=2, rqvae_codebook_size=2, n_epochs=1,
        d_model=8, d_ff=16, num_heads=2, d_kv=4, letter_base_vocab_size=8,
        num_enc_layers=1, num_dec_layers=1, max_len=2,
        dropout=0, n_beams=2, device="cpu", seed=7,
    )
    model.fit(evaluation.train_set)
    return model, evaluation


@pytest.mark.parametrize("scoring", ["beam", "exact"])
def test_letter_native_parity_collisions_and_cpu_fit(scoring, tmp_path):
    from cornac.metrics import Recall

    model, evaluation = tiny_letter()
    model.scoring = scoring
    cat = model.semantic_id_catalog()
    assert cat.address_to_items[(0, 0)] == (0, 1)
    assert cat.address_sids.shape == cat.tokenizer_sids.shape == (3, 2)
    inputs, mask = model._encoder_batch(torch, [[0]])
    model.model.eval()
    with torch.no_grad():
        if scoring == "beam":
            beams, scores = model.model.generate_beam(inputs, mask, 2, model.prefix_children)
            expected = np.full(3, -1e10)
            for sid, score in zip(beams, scores):
                expected[model.sid_to_item[sid]] = score
            raw = model.predict_semantic_ids([0], n_candidates=2)
            np.testing.assert_array_equal(raw.address_sids, beams)
            np.testing.assert_allclose(raw.scores, scores)
        else:
            expected = model.model.score_all_items(inputs, mask, torch.tensor(model.sid_table), model.scoring_batch_size)
            raw = model.predict_semantic_ids([0])
            oracle = prediction_from_catalog_scores(cat, expected)
            np.testing.assert_array_equal(raw.address_sids, oracle.address_sids)
            np.testing.assert_allclose(raw.scores, oracle.scores)
    np.testing.assert_allclose(model.score(0, [0]), expected)
    restored = model.load(model.save(str(tmp_path)))
    np.testing.assert_array_equal(restored.predict_semantic_ids([0]).address_sids, raw.address_sids)
    assert restored.semantic_id_catalog().stable_hash == cat.stable_hash
    # Avoid retraining here: the fit smoke above is sufficient; evaluate frozen model.
    model.trainable = False
    result, _ = evaluation.evaluate(model, [Recall(2)], user_based=False)
    assert "CCE/Recall@2" in result.metric_avg_results


def test_letter_raw_slots_and_budget(monkeypatch):
    model, _ = tiny_letter()
    outputs = [(0, 0), (8, 8), (0, 0)]
    budgets = []

    def decode(tokens, mask, budget, children):
        budgets.append(budget)
        return outputs, [-1., -2., -3.]

    monkeypatch.setattr(model.model, "generate_beam", decode)
    raw = model.predict_semantic_ids([0], n_candidates=5)
    assert budgets == [5]
    assert model.n_beams == 2
    assert raw.address_sids.tolist() == [list(sid) for sid in outputs]
    assert raw.catalog_validity.tolist() == [True, False, True]


def tiny_rpg(monkeypatch):
    from cornac.data import FeatureModality
    from cornac.eval_methods import SemanticIDEvaluation
    from cornac.models import RPG

    evaluation = SemanticIDEvaluation.from_splits(
        [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)],
        [("u", "te", "a", 3), ("u", "te", "b", 4)], fmt="USIT",
        item_feature=FeatureModality(features=np.eye(3), ids=["a", "b", "c"]),
    )
    model = RPG(
        n_codebook=2, codebook_size=2, n_epochs=1, d_model=8,
        n_layer=1, n_head=2, n_inner=16, max_len=2,
        resid_dropout=0, embd_dropout=0, attn_dropout=0,
        n_beams=2, graph_edges=2, propagation_steps=1, device="cpu", seed=7,
    )
    # faiss OPQ uses 8-bit PQ and needs >=256 training items; freeze only
    # tokenization to keep this real training/graph-building CPU smoke tiny.
    with monkeypatch.context() as patch:
        patch.setattr(RPG, "_opq_tokenize", lambda self, feats, train_mask: np.array([[0, 0], [0, 0], [1, 1]]))
        model.fit(evaluation.train_set)
    return model, evaluation


@pytest.mark.parametrize("scoring", ["graph", "exact"])
def test_rpg_native_parity_rng_and_persistence(monkeypatch, scoring, tmp_path):
    from cornac.models.rpg.rpg import graph_decode

    model, _ = tiny_rpg(monkeypatch)
    model.scoring = scoring
    state = model.graph_rng.get_state()
    model.model.eval()
    with torch.no_grad():
        tokens = torch.tensor([[model.pad_idx, 0]])
        logits = model.model.next_item_logits(tokens, (tokens != model.pad_idx).float())[0]
    if scoring == "graph":
        candidates, scores = graph_decode(logits.reshape(-1).numpy(), model.adjacency,
            model.item_tokens, 2, model.propagation_steps, model.graph_rng)
        after_native = model.graph_rng.get_state()
        expected = np.full(3, -1e10)
        expected[candidates] = scores
        model.graph_rng.set_state(state)
        raw = model.predict_semantic_ids([0])
        np.testing.assert_array_equal(raw.address_sids, model.sid_table[candidates])
        np.testing.assert_allclose(raw.scores, scores)
        assert all(np.array_equal(a, b) for a, b in zip(after_native, model.graph_rng.get_state()))
        assert not all(np.array_equal(a, b) for a, b in zip(state, after_native))
    else:
        expected = model._exact_item_scores(logits)[0].numpy().astype(float)
        raw = model.predict_semantic_ids([0])
        oracle = prediction_from_catalog_scores(model.semantic_id_catalog(), expected)
        np.testing.assert_array_equal(raw.address_sids, oracle.address_sids)
        np.testing.assert_allclose(raw.scores, oracle.scores)
        assert len(raw.scores) == 2  # maximum member score collapses the collision
    model.graph_rng.set_state(state)
    np.testing.assert_allclose(model.score(0, [0]), expected)
    restored = model.load(model.save(str(tmp_path)))
    # Save copied the RNG state, so compare the same next stochastic call.
    np.testing.assert_array_equal(restored.predict_semantic_ids([0]).address_sids,
                                  model.predict_semantic_ids([0]).address_sids)
    assert model.semantic_id_catalog().address_to_items[(0, 0)] == (0, 1)
    assert len(model.predict_semantic_ids([]).scores) == 0
    np.testing.assert_array_equal(model.score(0, []), np.ones(3))


def test_rpg_graph_order_collisions_and_budget(monkeypatch):
    import cornac.models.rpg.rpg as backend
    model, evaluation = tiny_rpg(monkeypatch)
    seen = []

    def decode(flat, adjacency, tokens, budget, steps, rng):
        assert rng is model.graph_rng
        seen.append(budget)
        return np.array([2, 0, 1]), np.array([-1., -2., -3.])

    with monkeypatch.context() as patch:
        patch.setattr(backend, "graph_decode", decode)
        raw = model.predict_semantic_ids([0], n_candidates=5)
    assert seen == [5]
    assert model.n_beams == 2
    assert raw.address_sids.tolist() == [[1, 1], [0, 0], [0, 0]]
    from cornac.metrics import Recall
    model.trainable = False
    result, _ = evaluation.evaluate(model, [Recall(2)], user_based=False)
    assert "CCE/Recall@2" in result.metric_avg_results


class ConstantCPD:
    n_digit = 2
    codebook_size = 2

    def decode_logits(self, memory, padding, partial):
        return torch.tensor([[[4., 0.], [3., 0.]]]).expand(len(partial), -1, -1)


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
def test_cpd_raw_single_batch_default_parity(diagnostics, constrained):
    from cornac.models.diffgrm.diffgrm import cpd_decode, cpd_decode_batch

    memory, padding = torch.zeros(2, 1, 4), torch.zeros(2, 1, dtype=torch.bool)
    kwargs = dict(beam_size=4, catalog_codes=np.array([[0, 0], [1, 1]]),
                  constrained=constrained, return_diagnostics=diagnostics)
    default = cpd_decode_batch(ConstantCPD(), memory, padding, **kwargs)
    extra = cpd_decode_batch(ConstantCPD(), memory, padding, return_raw=True, **kwargs)
    assert extra[:-1] == default
    raw = extra[-1]
    assert len(raw) == 2
    for entry in raw:
        assert len(entry["address_sids"]) == len(entry["scores"]) == len(entry["catalog_validity"]) == 4
        # Parallel assignment orders produce repeated complete paths.
        assert len(set(entry["address_sids"])) < len(entry["address_sids"])
        assert entry["catalog_validity"] == [
            sid in {(0, 0), (1, 1)} and np.isfinite(score)
            for sid, score in zip(entry["address_sids"], entry["scores"])
        ]
    single = cpd_decode(ConstantCPD(), memory[:1], padding[:1], return_raw=True, **kwargs)
    assert single == tuple(part[0] for part in extra)
    legacy_single = cpd_decode(ConstantCPD(), memory[:1], padding[:1], **kwargs)
    assert legacy_single == tuple(part[0] for part in default)
    with pytest.raises(ValueError, match="one encoded history"):
        cpd_decode(ConstantCPD(), memory, padding, **kwargs)


def test_cpd_raw_marks_nonfinite_and_invalid_slots():
    from cornac.models.diffgrm.diffgrm import cpd_decode
    memory, padding = torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.bool)
    codes, scores, raw = cpd_decode(ConstantCPD(), memory, padding,
        beam_size=16, catalog_codes=np.array([[1, 1]]), return_raw=True)
    assert len(raw["address_sids"]) > len(codes)
    assert any(not valid for valid in raw["catalog_validity"])
    for sid, score, valid in zip(raw["address_sids"], raw["scores"], raw["catalog_validity"]):
        assert valid == (sid == (1, 1) and -1 not in sid and np.isfinite(score))
    with pytest.raises(ValueError, match="permutation"):
        cpd_decode(ConstantCPD(), memory, padding, beam_size=2, digit_order=[0, 0])


def tiny_diffgrm():
    from cornac.eval_methods import SemanticIDEvaluation
    from cornac.models import DiffGRM
    evaluation = SemanticIDEvaluation.from_splits(
        [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)],
        [("u", "te", "a", 3), ("u", "te", "b", 4)], fmt="USIT")
    model = DiffGRM(
        item_sids=np.array([[0, 0], [0, 0], [1, 1]]), n_digit=2,
        codebook_size=2, d_model=8, encoder_n_layer=1, decoder_n_layer=1,
        n_head=2, n_inner=16, dropout=0, max_len=2, min_history=1,
        n_epochs=1, n_views=1, beam_size=4, lr_schedule="constant",
        device="cpu", seed=7,
    )
    model.fit(evaluation.train_set)
    return model, evaluation


@pytest.mark.parametrize("scoring", ["released", "paper", "catalog", "fixed"])
@pytest.mark.parametrize("policy", ["all", "first", "last"])
def test_diffgrm_native_parity_raw_and_persistence(scoring, policy, tmp_path):
    from cornac.models.diffgrm.diffgrm import cpd_decode

    model, _ = tiny_diffgrm()
    model.scoring = scoring
    model.collision_policy = policy
    model.model.eval()
    # Original master's left-aligned history input and decode options.
    with torch.no_grad():
        inputs = torch.tensor([[0, model.pad_idx]])
        memory, padding = model.model.encode_history(inputs, torch.tensor([[1., 0.]]))
        codes, scores, diagnostics, raw = cpd_decode(model.model, memory, padding,
            beam_size=model.beam_size, catalog_codes=model.sid_table,
            valid_code_set=model.sid_to_items_, constrained=scoring == "catalog",
            digit_order=model.fixed_decode_order_ if scoring == "fixed" else None,
            greedy_final=scoring == "released", return_diagnostics=True, return_raw=True)
    expected = np.full(3, -1e10)
    for sid, score in zip(codes, scores):
        members = model.sid_to_items_[sid]
        members = members[:1] if policy == "first" else members[-1:] if policy == "last" else members
        expected[members] = score
    np.testing.assert_allclose(model.score(0, [0]), expected)
    pred = model.predict_semantic_ids(np.array([0]))
    np.testing.assert_array_equal(pred.address_sids, raw["address_sids"])
    np.testing.assert_allclose(pred.scores, raw["scores"])
    assert pred.catalog_validity.tolist() == raw["catalog_validity"]
    assert model.semantic_id_catalog().address_to_items[(0, 0)] == (0, 1)
    restored = model.load(model.save(str(tmp_path)))
    np.testing.assert_allclose(restored.score(0, [0]), expected)
    np.testing.assert_array_equal(restored.predict_semantic_ids([0]).address_sids, pred.address_sids)
    assert len(model.predict_semantic_ids([]).scores) == 0


def test_diffgrm_requested_budget_and_evaluation(monkeypatch):
    import cornac.models.diffgrm.diffgrm as backend
    from cornac.metrics import Recall

    model, evaluation = tiny_diffgrm()
    original = backend.cpd_decode
    seen = []

    def decode(*args, **kwargs):
        seen.append((kwargs["beam_size"], kwargs["return_raw"]))
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(backend, "cpd_decode", decode)
        raw = model.predict_semantic_ids([0], n_candidates=8)
    assert seen == [(8, True)]
    assert model.beam_size == 4
    assert len(raw.scores) == 8
    model.trainable = False
    result, _ = evaluation.evaluate(model, [Recall(2)], user_based=False)
    assert "CCE/Recall@2" in result.metric_avg_results


@pytest.mark.parametrize("kind", ["tiger", "letter", "rpg", "diffgrm"])
def test_fitted_catalog_cached_refit_and_old_pickle(kind, monkeypatch, tmp_path):
    if kind == "tiger":
        # Full fitted dataset is needed below; reuse the known tiny fixture.
        from cornac.data import FeatureModality
        from cornac.eval_methods import SemanticIDEvaluation
        evaluation = SemanticIDEvaluation.from_splits(
            [("u", "tr", "a", 0), ("u", "tr", "b", 1), ("u", "tr", "c", 2)],
            [("u", "te", "a", 3), ("u", "te", "b", 4)], fmt="USIT",
            item_feature=FeatureModality(features=np.eye(3), ids=["a", "b", "c"]))
        model = tiny_tiger()
        model.rqvae_n_epochs = 1
        model.rqvae_hidden_dims = (4,)
        model.rqvae_latent_dim = 2
        model.fit(evaluation.train_set)
    elif kind == "letter":
        model, evaluation = tiny_letter()
    elif kind == "rpg":
        model, evaluation = tiny_rpg(monkeypatch)
    else:
        model, evaluation = tiny_diffgrm()
    first = model.semantic_id_catalog()
    assert model.semantic_id_catalog() is first
    model.predict_semantic_ids([0])
    assert model.semantic_id_catalog() is first
    model.trainable = False
    with pytest.warns(UserWarning, match="already fitted"):
        model.fit(evaluation.train_set)
    second = model.semantic_id_catalog()
    assert second is not first
    assert second.stable_hash == first.stable_hash
    new_codes = np.array([[1, 0], [0, 1], [1, 0]])
    model.trainable = True
    with monkeypatch.context() as patch:
        if kind in ("tiger", "letter"):
            patch.setattr(type(model), "_tokenize", lambda self, torch, feats: new_codes.copy())
        elif kind == "rpg":
            patch.setattr(type(model), "_opq_tokenize", lambda self, feats, train_mask: new_codes.copy())
        else:
            model.item_sids = new_codes
        with pytest.warns(UserWarning, match="already fitted"):
            model.fit(evaluation.train_set)
    third = model.semantic_id_catalog()
    assert third is model.semantic_id_catalog()
    assert third.stable_hash != second.stable_hash
    np.testing.assert_array_equal(third.tokenizer_sids, new_codes)
    if kind in ("tiger", "letter"):
        model._build_semantic_ids(np.array([[0, 1], [0, 0], [1, 1]]))
        rebuilt = model.semantic_id_catalog()
        assert rebuilt is not third
        assert rebuilt.stable_hash != third.stable_hash
    expected_hash = model.semantic_id_catalog().stable_hash
    # Simulate an older model pickle that has SID state but no cached contract.
    del model.semantic_id_catalog_
    restored = model.load(model.save(str(tmp_path)))
    old_state_catalog = restored.semantic_id_catalog()
    assert old_state_catalog.stable_hash == expected_hash
    assert restored.semantic_id_catalog() is old_state_catalog
