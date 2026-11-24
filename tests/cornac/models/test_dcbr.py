# Copyright 2026 The Cornac Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Small CPU checks of DCBR."""

import numpy as np
import pytest
import scipy.sparse as sp

from cornac.models import DCBR

from ._bundle_checks import check_bundle_model, toy_evaluation


def tiny(**kwargs):
    params = dict(
        emb_size=8,
        num_layers=1,
        ub_layer_weights=(0.5, 0.5),
        ui_layer_weights=(0.5, 0.5),
        bi_layer_weights=(0.5, 0.5),
        diff_steps=3,
        diff_dims=(16,),
        diff_batch_size=5,
        num_epochs=4,
        batch_size=16,
        learning_rate=0.05,
        # Keep BPR observable on a tiny catalog after summing all CL terms.
        c_lambda=0.01,
        eval_every=2,
        eval_k=5,
        device="cpu",
        seed=3,
    )
    params.update(kwargs)
    return DCBR(**params)


@pytest.fixture(scope="module")
def evaluation():
    pytest.importorskip("torch")
    return toy_evaluation()


def test_contract(evaluation):
    model = check_bundle_model(tiny, evaluation)
    assert model.U.shape[1] == 8  # UI and BI views fused


@pytest.mark.parametrize("eval_every, epochs", [(2, [2, 3]), (100, [3])])
def test_validation_includes_final_epoch(evaluation, eval_every, epochs):
    model = tiny(num_epochs=3, eval_every=eval_every)
    model.fit(evaluation.train_set, evaluation.val_set)
    assert [entry[0] for entry in model.val_history] == epochs
    assert model.best_epoch in epochs


def test_untrainable_fit_requires_fitted_vectors(evaluation):
    with pytest.raises(ValueError, match="fitted scoring vectors"):
        tiny(trainable=False).fit(evaluation.train_set)


def test_diffusion_rebuilds_top_k_graph(evaluation):
    import torch

    model = tiny(rebuild_k=2)
    model.rng = np.random.RandomState(0)
    torch.manual_seed(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    before = module.ub_graph.to_dense().clone()
    module.epoch_start(0)
    after = module.ub_graph.to_dense()
    nu = module.num_users
    links = (after[:nu, nu:] > 0).sum(dim=1)
    assert torch.all(links == 2)  # exactly rebuild_k bundles per user
    assert not torch.equal(before, after)
    assert torch.all(torch.diagonal(after) > 0)  # self-loops


def test_diffusion_schedule_and_denoise_shapes():
    torch = pytest.importorskip("torch")
    from cornac.models.dcbr.dcbr import Diffusion

    diffusion = Diffusion(
        num_bundles=6,
        steps=4,
        noise_scale=0.1,
        noise_min=1e-4,
        noise_max=0.02,
        hidden_dims=(8,),
        time_emb_size=5,
        dropout=0.0,
    )
    assert torch.all(diffusion.sqrt_alpha_bar[1:] <= diffusion.sqrt_alpha_bar[:-1])
    x0 = torch.zeros(3, 6)
    x0[0, 1] = 1.0
    loss, x0_hat = diffusion.training_loss(x0)
    assert x0_hat.shape == (3, 6) and torch.isfinite(loss)
    assert diffusion.denoise(x0).shape == (3, 6)


def test_diffusion_not_in_main_optimizer(evaluation):
    import torch

    model = tiny()
    model.rng = np.random.RandomState(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    main = {
        id(p)
        for group in model._optimizer(module).param_groups
        for p in group["params"]
    }
    assert not main & {id(p) for p in module.diffusion.parameters()}


def test_argument_validation():
    with pytest.raises(ValueError):
        DCBR(num_layers=1)
    with pytest.raises(ValueError):
        DCBR(rebuild_k=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"emb_size": 0},
        {"batch_size": 0},
        {"eval_every": 0},
        {"patience": 0},
        {"diff_steps": 1.5},
        {"diff_dims": ()},
        {"diff_dims": (0,)},
        {"time_emb_size": 1},
        {"diff_batch_size": 0},
        {"c_temp": 0},
        {"diff_lr": float("nan")},
        {"noise_scale": 0},
        {"noise_min": 0.03},
        {"noise_max": float("inf")},
        {"diff_dropout": 1},
        {"diff_norm": 1},
        {"gammas": (-1, 0)},
        {"view_eps": (0, float("nan"), 0)},
        {"ub_layer_weights": (0, 0, 0)},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DCBR(**kwargs)


def test_diffusion_schedule_and_weighted_objective(monkeypatch):
    torch = pytest.importorskip("torch")
    from cornac.models.dcbr.dcbr import Diffusion

    diffusion = Diffusion(3, 4, 0.1, 1e-4, 0.02, (8,), 4, 0.0)
    variance = 0.1 * np.linspace(1e-4, 0.02, 4)
    betas = 1 - (1 - variance) / np.r_[1, (1 - variance)[:-1]]
    betas[0] = 1e-4
    cumulative = np.cumprod(1 - betas)
    np.testing.assert_allclose(diffusion.betas.numpy(), betas)
    np.testing.assert_allclose(diffusion.sqrt_alpha_bar.numpy(), np.sqrt(cumulative))

    timesteps = torch.tensor([0, 1, 3])
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: timesteps)
    x0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    predicted = torch.zeros_like(x0, requires_grad=True)
    monkeypatch.setattr(diffusion.denoiser, "forward", lambda *args: predicted)
    loss, _ = diffusion.training_loss(x0)
    snr = cumulative / (1 - cumulative)
    expected = np.mean([1, snr[0] - snr[1], snr[2] - snr[3]]) / 3
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()


def test_denoiser_input_normalization_and_deterministic_sampling():
    torch = pytest.importorskip("torch")
    from cornac.models.dcbr.dcbr import Denoiser, Diffusion

    for norm in (False, True):
        denoiser = Denoiser(3, (8, 4), 5, 0.0, norm=norm)
        seen = []
        handle = denoiser.encoder[0].register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach())
        )
        rows = torch.tensor([[3.0, 4.0, 0.0]])
        assert denoiser(rows, torch.tensor([1])).shape == (1, 3)
        expected = rows / 5 if norm else rows
        torch.testing.assert_close(seen[0][:, :3], expected)
        handle.remove()
    diffusion = Diffusion(3, 1, 0.1, 1e-4, 0.02, (8,), 4, 0.9)
    # Reconstruction must disable dropout even when the module is in train mode.
    torch.testing.assert_close(diffusion.denoise(rows), diffusion.denoise(rows))


def test_consistency_uses_raw_detached_embeddings(evaluation, monkeypatch):
    torch = pytest.importorskip("torch")
    from cornac.models.dcbr.dcbr import DCBRModule

    model = tiny(num_epochs=1)
    model.rng = np.random.RandomState(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    seen = []

    def check(rows, users, bundles):
        seen.append(True)
        assert not users.requires_grad and not bundles.requires_grad
        assert bundles.data_ptr() == module.bundles.data_ptr()
        for user in users:
            assert torch.any(torch.all(module.users.detach() == user, dim=1))
        return DCBRModule.consistency_loss(rows, users, bundles)

    monkeypatch.setattr(module, "consistency_loss", check)
    monkeypatch.setattr(
        module, "fused", lambda **kwargs: pytest.fail("BLCC used fused embeddings")
    )
    module.epoch_start(0)
    assert seen and module.users.grad is None and module.bundles.grad is None


def test_contrastive_loss_sums_all_terms(evaluation, monkeypatch):
    torch = pytest.importorskip("torch")
    from scipy.special import logsumexp

    model = tiny(lambda_reg=0)
    model.rng = np.random.RandomState(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    values = {
        "UB": (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        ),
        "UI": (
            torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        ),
        "BI": (
            torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        ),
    }
    monkeypatch.setattr(module, "views", lambda noise: values)
    index = torch.tensor([0, 1])
    _, parts = module.loss(index, index, index.flip(0))

    def nce(a, b):
        a, b = a.numpy(), b.numpy()
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        logits = a @ b.T / model.c_temp
        return np.mean(logsumexp(logits, axis=1) - np.diag(logits))

    inter = sum(
        nce(values[a][k], values[b][k])
        for a, b in (("UB", "UI"), ("UB", "BI"), ("UI", "BI"))
        for k in (0, 1)
    )
    intra = sum(nce(values[a][k], values[a][k]) for a in values for k in (0, 1))
    assert parts["contrastive"] == pytest.approx(
        model.gammas[0] * inter + model.gammas[1] * intra
    )


def test_aggregation_noise_uses_relation_type(evaluation, monkeypatch):
    torch = pytest.importorskip("torch")
    import cornac.models.dcbr.dcbr as backend

    model = tiny(view_eps=(0.1, 0.2, 0.3))
    model.rng = np.random.RandomState(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    noise = []
    monkeypatch.setattr(backend, "perturb", lambda x, eps: noise.append(eps) or x)
    module.views(noise=True)
    assert noise == [0.3, 0.2]
    noise.clear()
    module.views(noise=False)
    assert noise == [0, 0]


def test_scoring_ignores_rebuilt_ub_graph(evaluation):
    torch = pytest.importorskip("torch")
    model = tiny()
    model.rng = np.random.RandomState(0)
    module = model._build_module(evaluation.train_set, torch.device("cpu"))
    before = module.embeddings()
    module.ub_graph = None
    after = module.embeddings()
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b)


def test_dataset_type_and_saturated_users(evaluation):
    import copy

    from cornac.data import Dataset

    with pytest.raises(ValueError, match="BundleDataset"):
        tiny().fit(Dataset.from_uir([("u", "b", 1.0)]))
    saturated = copy.deepcopy(evaluation.train_set)
    matrix = saturated.csr_matrix
    with pytest.warns(sp.SparseEfficiencyWarning):
        matrix[:] = sp.csr_matrix(np.ones(matrix.shape))
    with pytest.raises(ValueError, match="unobserved bundle"):
        tiny().fit(saturated)


def test_frozen_fit_validates_catalog(evaluation):
    import copy

    model = tiny(num_epochs=0).fit(evaluation.train_set)
    original = model.score(0).copy()
    model.trainable = False
    assert model.fit(evaluation.train_set, evaluation.val_set) is model
    np.testing.assert_array_equal(model.score(0), original)
    reordered = copy.copy(evaluation.train_set)
    reordered.uid_map = dict(reversed(list(reordered.uid_map.items())))
    # Swap mapping values, since dict equality ignores insertion order.
    ids = list(reordered.uid_map)
    reordered.uid_map[ids[0]], reordered.uid_map[ids[1]] = (
        reordered.uid_map[ids[1]],
        reordered.uid_map[ids[0]],
    )
    with pytest.raises(ValueError, match="ordered ID mappings"):
        model.fit(reordered)
    np.testing.assert_array_equal(model.score(0), original)
    model.V = model.V[:-1]
    with pytest.raises(ValueError, match="dimensions"):
        model.fit(evaluation.train_set)
