# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""CPU regression tests for RecVAE training and scoring."""

import numpy as np
import pytest

from cornac.eval_methods import BaseMethod, ranking_eval
from cornac.metrics import NDCG, Recall
from cornac.models import RecVAE


torch = pytest.importorskip("torch")


@pytest.fixture
def evaluation():
    return BaseMethod.from_splits(
        train_data=[(str(u), str(u + offset), 1.0)
                    for u in range(8) for offset in range(0, 128, 8)],
        test_data=[(str(u), str((u + 1) % 8 + 24), 1.0) for u in range(8)],
        seed=7,
    )


@pytest.fixture(params=[False, True])
def model(request):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield RecVAE(
        hidden_dim=16, latent_dim=4, batch_size=8, n_epochs=2,
        n_enc_epochs=1, n_dec_epochs=1, not_alternating=request.param,
        seed=7, use_gpu=False,
    )
    torch.set_num_threads(old_threads)


def test_scoring_is_deterministic_without_consuming_rng(model, evaluation):
    model.fit(evaluation.train_set)
    rng_state = torch.get_rng_state().clone()
    scores = model.score(0)

    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(model.score(0), scores)
    assert model.score(0, 3) == scores[3]

    ratings = torch.tensor(evaluation.train_set.matrix[0, :].toarray(), dtype=torch.float32)
    with torch.no_grad():
        mu, _ = model.recvae_model.encoder(ratings, dropout_rate=0)
        expected = model.recvae_model.decoder(mu).numpy().flatten()
    np.testing.assert_array_equal(scores, expected)

    metrics = [NDCG(k=100), Recall(k=100)]
    first = ranking_eval(model, metrics, evaluation.train_set, evaluation.test_set)
    second = ranking_eval(model, metrics, evaluation.train_set, evaluation.test_set)
    assert first == second
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_training_resumes_after_scoring(model, evaluation, monkeypatch):
    from cornac.models.recvae.recvae import VAE

    calls = []
    original_forward = VAE.forward

    def record_forward(self, *args, **kwargs):
        calculate_loss = kwargs.get("calculate_loss", True)
        result = original_forward(self, *args, **kwargs)
        output = result[1] if calculate_loss else result
        calls.append((calculate_loss, self.training, torch.is_grad_enabled(),
                      output.requires_grad, bool(torch.isfinite(output).all())))
        return result

    monkeypatch.setattr(VAE, "forward", record_forward)
    model.fit(evaluation.train_set)

    training_calls = [call for call in calls if call[0]]
    scoring_calls = [call for call in calls if not call[0]]
    assert len(training_calls) == (2 if model.not_alternating else 4)
    assert len(scoring_calls) == 2 * evaluation.train_set.num_users
    assert all(call == (True, True, True, True, True) for call in training_calls)
    assert all(call == (False, False, False, False, True) for call in scoring_calls)
    first_score = next(i for i, call in enumerate(calls) if not call[0])
    assert any(call[0] for call in calls[first_score + 1:])

    model.score(0)
    before = model.recvae_model.encoder.fc1.weight.detach().clone()
    optimizer = torch.optim.Adam(model.recvae_model.parameters(), lr=model.lr)
    model.run(model.recvae_model, [optimizer], evaluation.train_set,
              model.batch_size, 1, model.beta, model.gamma, 0.5)
    assert calls[-1] == (True, True, True, True, True)
    assert not torch.equal(before, model.recvae_model.encoder.fc1.weight)
    np.testing.assert_array_equal(model.score(0), model.score(0))
