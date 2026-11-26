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
"""PyTorch module of DCBR (imported only inside ``fit``)."""

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..crosscbr.crosscbr import graph_tensor


def binary(matrix):
    """Binary float64 CSR copy of a sparse matrix (None stays None)."""
    if matrix is None:
        return None
    out = (sp.csr_matrix(matrix) != 0).astype(np.float64).tocsr()
    out.eliminate_zeros()
    return out


def user_item_matrix(train_set):
    """Training user x constituent item matrix of a :obj:`BundleDataset`.

    Uses the supplied ``user_item_matrix`` when available, otherwise derives
    it from the user's training bundles (UB @ BI, binarized). Only training
    observations are used, so no validation or test signal can leak.
    """
    if train_set.user_item_matrix is not None:
        return binary(train_set.user_item_matrix)
    return binary(binary(train_set.csr_matrix) @ binary(train_set.bundle_item_matrix))


def _observed(keys, query):
    positions = np.searchsorted(keys, query)
    hit = positions < len(keys)
    hit[hit] = keys[positions[hit]] == query[hit]
    return hit


def sample_negatives(positives, users, rng, keys, max_tries=100):
    """Uniform negatives: one column of ``positives`` not observed for each user.

    ``keys`` contains sorted user * num_cols + bundle IDs, built once per fit.
    Vectorized rejection sampling uses binary searches against these keys.
    """
    num_cols = positives.shape[1]
    users = np.asarray(users, dtype=np.int64)
    if np.any(positives.indptr[users + 1] - positives.indptr[users] == num_cols):
        raise ValueError(
            "each training user needs an unobserved bundle for BPR sampling"
        )
    negatives = rng.randint(num_cols, size=len(users))
    for _ in range(max_tries):
        hit = _observed(keys, users * num_cols + negatives)
        if not hit.any():
            break
        negatives[hit] = rng.randint(num_cols, size=int(hit.sum()))
    # Rejection sampling can exhaust its budget for dense users. Finish with an
    # exact uniform draw from their complement rather than returning positives.
    hit = _observed(keys, users * num_cols + negatives)
    for user in np.unique(users[hit]):
        observed = positives.indices[
            positives.indptr[user] : positives.indptr[user + 1]
        ]
        candidates = np.setdiff1d(np.arange(num_cols), observed, assume_unique=True)
        mask = hit & (users == user)
        negatives[mask] = rng.choice(candidates, size=int(mask.sum()))
    return negatives


def sp_to_torch(matrix, device=None):
    """SciPy sparse matrix -> coalesced float32 torch sparse COO tensor."""
    coo = sp.coo_matrix(matrix)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    tensor = torch.sparse_coo_tensor(
        indices, values, coo.shape, check_invariants=False
    ).coalesce()
    return tensor if device is None else tensor.to(device)


def perturb(h, eps):
    """Noise augmentation h + sign(h) * eps * unit-norm uniform noise (SimGCL style)."""
    if eps <= 0:
        return h
    noise = F.normalize(torch.rand_like(h), dim=-1)
    return h + torch.sign(h) * noise * eps


def lightgcn_propagate(
    graph, x, num_layers, layer_weights=None, normalize=False, noise_eps=0.0
):
    """LightGCN propagation without feature transforms or nonlinearities.

    Parameters
    ----------
    graph: torch sparse tensor
        Normalized (n x n) adjacency.
    x: torch.Tensor
        Layer-0 embeddings (n x d).
    num_layers: int
        Number of propagation steps.
    layer_weights: sequence of float, optional
        Weights of layers 0..num_layers. Default: plain sum.
    normalize: bool
        L2-normalize each propagated layer before combining (layer 0 is kept raw),
        as in CrossCBR and MultiCBR.
    noise_eps: float
        Noise magnitude added after every propagation step (MultiCBR).

    Returns
    -------
    torch.Tensor of shape (n x d)
    """
    layers = [x]
    h = x
    for _ in range(num_layers):
        h = torch.sparse.mm(graph, h)
        h = perturb(h, noise_eps)
        layers.append(F.normalize(h, p=2, dim=1) if normalize else h)
    if layer_weights is None:
        return torch.stack(layers, dim=0).sum(dim=0)
    if len(layer_weights) != num_layers + 1:
        raise ValueError("layer_weights must have num_layers + 1 values")
    return sum(w * layer for w, layer in zip(layer_weights, layers))


def bpr_loss(pos_scores, neg_scores):
    """Mean BPR loss -log sigmoid(pos - neg) = softplus(neg - pos)."""
    return F.softplus(neg_scores - pos_scores).mean()


def info_nce(a, b, temperature):
    """InfoNCE over a batch: row i of ``a`` is positive with row i of ``b``.

    Cosine similarities divided by ``temperature``; every row of ``b``
    (including the positive) is in the denominator. Returns the mean loss.
    """
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    logits = a @ b.T / temperature
    targets = torch.arange(a.shape[0], device=a.device)
    return F.cross_entropy(logits, targets)


class Denoiser(nn.Module):
    """MLP that predicts a clean user-bundle row x_0 from a noised row x_t and step t."""

    def __init__(self, num_bundles, hidden_dims, time_emb_size, dropout, norm=False):
        super().__init__()
        self.time_emb_size = time_emb_size
        self.norm = norm
        self.time_proj = nn.Linear(time_emb_size, time_emb_size)
        dims = [num_bundles + time_emb_size, *reversed(hidden_dims)]
        self.encoder = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        out_dims = [*hidden_dims, num_bundles]
        self.decoder = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(out_dims[:-1], out_dims[1:])
        )
        self.dropout = nn.Dropout(dropout)
        for layer in (self.time_proj, *self.encoder, *self.decoder):
            nn.init.normal_(layer.weight, std=math.sqrt(2 / sum(layer.weight.shape)))
            nn.init.normal_(layer.bias, std=0.001)

    def time_embedding(self, t):
        half = self.time_emb_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.time_emb_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.time_proj(emb)

    def forward(self, x, t, use_dropout=True):
        if self.norm:
            x = F.normalize(x, dim=-1)
        if use_dropout:
            x = self.dropout(x)
        h = torch.cat([x, self.time_embedding(t)], dim=-1)
        for layer in self.encoder:
            h = torch.tanh(layer(h))
        for k, layer in enumerate(self.decoder):
            h = layer(h)
            if k < len(self.decoder) - 1:
                h = torch.tanh(h)
        return h


class Diffusion(nn.Module):
    """Gaussian diffusion over user-bundle rows (x_0 prediction), DiffRec style."""

    def __init__(
        self,
        num_bundles,
        steps,
        noise_scale,
        noise_min,
        noise_max,
        hidden_dims,
        time_emb_size,
        dropout,
        norm=False,
    ):
        super().__init__()
        self.steps = steps
        self.denoiser = Denoiser(num_bundles, hidden_dims, time_emb_size, dropout, norm)
        variance = noise_scale * np.linspace(
            noise_min, noise_max, steps, dtype=np.float64
        )
        alpha_bar = 1.0 - variance
        betas = 1.0 - alpha_bar / np.append(1.0, alpha_bar[:-1])
        betas = np.minimum(betas, 0.999)
        betas[0] = 1e-4
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)
        alpha_bar_prev = np.append(1.0, alpha_bar[:-1])
        buffers = {
            "betas": betas,
            "snr": alpha_bar / (1.0 - alpha_bar),
            "sqrt_alpha_bar": np.sqrt(alpha_bar),
            "sqrt_one_minus_alpha_bar": np.sqrt(1.0 - alpha_bar),
            "post_coef_x0": betas * np.sqrt(alpha_bar_prev) / (1.0 - alpha_bar),
            "post_coef_xt": (1.0 - alpha_bar_prev)
            * np.sqrt(alphas)
            / (1.0 - alpha_bar),
        }
        for name, value in buffers.items():
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))

    def q_sample(self, x0, t, noise):
        return (
            self.sqrt_alpha_bar[t].to(x0.dtype)[:, None] * x0
            + self.sqrt_one_minus_alpha_bar[t].to(x0.dtype)[:, None] * noise
        )

    def training_loss(self, x0):
        t = torch.randint(0, self.steps, (x0.shape[0],), device=x0.device)
        x0_hat = self.denoiser(self.q_sample(x0, t, torch.randn_like(x0)), t)
        weights = torch.where(t == 0, 1.0, self.snr[t - 1] - self.snr[t])
        mse = ((x0_hat - x0) ** 2).mean(dim=1)
        return (weights * mse).mean(), x0_hat

    @torch.no_grad()
    def denoise(self, x0):
        """Reverse process from the clean row (no added noise), posterior means."""
        x = x0
        for step in reversed(range(self.steps)):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            x0_hat = self.denoiser(x, t, use_dropout=False)
            x = (
                (
                    self.post_coef_x0[step].to(x.dtype) * x0_hat
                    + self.post_coef_xt[step].to(x.dtype) * x
                )
                if step
                else x0_hat
            )
        return x


class DCBRModule(nn.Module):
    """UI and BI views (scored) plus a UB view learned on a diffusion-rebuilt
    user-bundle graph, tied together by inter- and intra-view InfoNCE."""

    def __init__(
        self,
        ui,
        ub,
        bi,
        emb_size,
        num_layers,
        layer_weights,
        view_eps,
        omega,
        c_lambda,
        c_temp,
        gammas,
        lambda_reg,
        diff_steps,
        noise_scale,
        noise_min,
        noise_max,
        diff_dims,
        time_emb_size,
        diff_dropout,
        diff_lr,
        diff_batch_size,
        blcc_weight,
        rebuild_k,
        rng,
        device,
        diff_norm=False,
    ):
        super().__init__()
        self.ub = ub.tocsr()
        self.num_users, self.num_items = ui.shape
        self.num_bundles = ub.shape[1]
        self.num_layers, self.layer_weights, self.view_eps = (
            num_layers,
            layer_weights,
            view_eps,
        )
        self.omega, self.c_lambda, self.c_temp = omega, c_lambda, c_temp
        self.gammas, self.lambda_reg = gammas, lambda_reg
        self.diff_batch_size, self.blcc_weight = diff_batch_size, blcc_weight
        self.rebuild_k, self.rng, self.device = rebuild_k, rng, device

        self.users = nn.Parameter(torch.empty(self.num_users, emb_size))
        self.bundles = nn.Parameter(torch.empty(self.num_bundles, emb_size))
        self.items = nn.Parameter(torch.empty(self.num_items, emb_size))
        for weight in (self.users, self.bundles, self.items):
            nn.init.xavier_normal_(weight)

        self.ui_graph = graph_tensor(ui, device, rng)
        self.bi_graph = graph_tensor(bi, device, rng)
        self.bundle_agg = graph_tensor(bi, device, rng, bipartite=False)
        self.user_agg = graph_tensor(ui, device, rng, bipartite=False)
        if self.c_lambda:
            self.diffusion = Diffusion(
                self.num_bundles,
                diff_steps,
                noise_scale,
                noise_min,
                noise_max,
                diff_dims,
                time_emb_size,
                diff_dropout,
                norm=diff_norm,
            ).to(device)
            self.diff_optimizer = torch.optim.Adam(
                self.diffusion.parameters(), lr=diff_lr
            )
            self.ub_graph = self._ub_graph(self.ub)

    def embedding_parameters(self):
        return (self.users, self.bundles, self.items)

    def _ub_graph(self, ub):
        # The rebuilt user-bundle graph includes a self-loop at every node.
        users, bundles = ub.shape
        graph = sp.bmat([[sp.eye(users), ub], [ub.T, sp.eye(bundles)]], format="csr")
        degree = np.asarray(graph.sum(axis=1)).ravel()
        scale = sp.diags(1.0 / (np.sqrt(degree) + 1e-8))
        return sp_to_torch(scale @ graph @ scale, self.device)

    def _rows(self, users):
        return torch.as_tensor(
            self.ub[users].toarray(), dtype=torch.float32, device=self.device
        )

    def epoch_start(self):
        """Train the diffusion model one pass, then rebuild the UB view's graph."""
        frozen_u, frozen_b = self.users.detach(), self.bundles.detach()
        self.diffusion.train()
        order = self.rng.permutation(self.num_users)
        for start in range(0, self.num_users, self.diff_batch_size):
            users = order[start : start + self.diff_batch_size]
            x0 = self._rows(users)
            loss, x0_hat = self.diffusion.training_loss(x0)
            if self.blcc_weight:
                loss = loss + self.blcc_weight * self.consistency_loss(
                    x0_hat,
                    frozen_u[torch.as_tensor(users, device=self.device)],
                    frozen_b,
                )
            self.diff_optimizer.zero_grad()
            loss.backward()
            self.diff_optimizer.step()
        self.diffusion.eval()
        rows, cols = [], []
        for start in range(0, self.num_users, self.diff_batch_size):
            users = np.arange(start, min(start + self.diff_batch_size, self.num_users))
            scores = self.diffusion.denoise(self._rows(users))
            top = torch.topk(
                scores, min(self.rebuild_k, self.num_bundles), dim=1
            ).indices
            rows.append(np.repeat(users, top.shape[1]))
            cols.append(top.cpu().numpy().ravel())
        rebuilt = sp.csr_matrix(
            (
                np.ones(sum(len(r) for r in rows)),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=self.ub.shape,
        )
        self.ub_graph = self._ub_graph((rebuilt > 0).astype(np.float64))

    @staticmethod
    def consistency_loss(predicted_rows, users, bundles):
        """Align reconstructed bundles with detached raw embedding tables."""
        return F.mse_loss(predicted_rows.T @ users.detach(), bundles.detach())

    def _view(self, graph, left, right, name, noise):
        eps = self.view_eps[name] if noise else 0.0
        feats = lightgcn_propagate(
            graph,
            torch.cat([left, right]),
            self.num_layers,
            layer_weights=self.layer_weights[name],
            normalize=True,
            noise_eps=eps,
        )
        return feats[: left.shape[0]], feats[left.shape[0] :], eps

    def views(self, noise, include_ub=True):
        views = {}
        if include_ub and self.c_lambda:
            u_ub, b_ub, _ = self._view(
                self.ub_graph, self.users, self.bundles, "UB", noise
            )
            views["UB"] = (u_ub, b_ub)
        u_ui, i_ui, _ = self._view(self.ui_graph, self.users, self.items, "UI", noise)
        b_ui = perturb(
            torch.sparse.mm(self.bundle_agg, i_ui),
            self.view_eps["BI"] if noise else 0.0,
        )
        b_bi, i_bi, _ = self._view(self.bi_graph, self.bundles, self.items, "BI", noise)
        u_bi = perturb(
            torch.sparse.mm(self.user_agg, i_bi), self.view_eps["UI"] if noise else 0.0
        )
        views.update({"UI": (u_ui, b_ui), "BI": (u_bi, b_bi)})
        return views

    def fused(self, noise, views=None):
        views = self.views(noise, include_ub=False) if views is None else views
        (u_ui, b_ui), (u_bi, b_bi) = views["UI"], views["BI"]
        return (
            self.omega * u_ui + (1 - self.omega) * u_bi,
            self.omega * b_ui + (1 - self.omega) * b_bi,
        )

    def loss(self, users, pos, neg):
        views = self.views(noise=True)
        u, b = self.fused(noise=True, views=views)
        bpr = bpr_loss((u[users] * b[pos]).sum(1), (u[users] * b[neg]).sum(1))
        reg = sum(p.pow(2).sum() for p in self.embedding_parameters())
        total = bpr + self.lambda_reg * reg
        parts = {"bpr": bpr.item()}
        if self.c_lambda:
            names = ("UB", "UI", "BI")
            inter = [
                info_nce(views[a][k][idx], views[c][k][idx], self.c_temp)
                for k, idx in ((0, users), (1, pos))
                for i, a in enumerate(names)
                for c in names[i + 1 :]
            ]
            intra = [
                info_nce(views[a][k][idx], views[a][k][idx], self.c_temp)
                for k, idx in ((0, users), (1, pos))
                for a in names
            ]
            c_loss = self.gammas[0] * sum(inter) + self.gammas[1] * sum(intra)
            total = total + self.c_lambda * c_loss
            parts["contrastive"] = c_loss.item()
        return total, parts

    def embeddings(self):
        return self.fused(noise=False)
