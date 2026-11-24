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

from numbers import Integral

import numpy as np
from tqdm.auto import trange

from ..recommender import MEASURE_DOT, BundleRecommender
from ...exception import ScoreException
from ...utils import get_rng
from .graph import binary, sample_negatives, user_item_matrix

VIEWS = ("UB", "UI", "BI")


class DCBR(BundleRecommender):
    """DCBR: Disentangled Contrastive Bundle Recommendation with Conditional Diffusion.

    Users and bundles are scored from two LightGCN views fused by ``omega``:
    the user-item view (bundles as the mean of their items) and the
    bundle-item view (users as the mean of their items), each with fixed layer
    weights and noise augmentation. A third, user-bundle view is learned on a
    graph rebuilt every epoch by a diffusion model: an MLP denoiser trained on
    the training user-bundle rows (x_0 prediction, plus ``blcc_weight`` times a
    bundle-level consistency term), whose denoised rows keep the top
    ``rebuild_k`` bundles of each user. That view enters only the inter- and
    intra-view InfoNCE losses; training uses BPR plus L2 on all embeddings.

    Parameters
    ----------
    name: str, default: 'DCBR'

    emb_size: int, default: 64

    num_layers: int, default: 2

    ub_layer_weights, ui_layer_weights, bi_layer_weights: tuple of floats
        Weights of layers 0..num_layers of each view.

    view_eps: tuple of 3 floats, default: (0.05, 0.0, 0.2)
        Noise magnitude of the UB, UI and BI views (training only).

    omega: float, default: 0.5
        Weight of the UI view in the fused (scored) representation.

    c_lambda: float, default: 0.3
        Weight of the contrastive loss.

    c_temp: float, default: 0.08
        InfoNCE temperature.

    gammas: (float, float), default: (0.3, 0.2)
        Weights of the inter-view and intra-view terms.

    lambda_reg: float, default: 1e-6
        L2 weight on all embedding parameters (no Adam weight decay).

    diff_steps: int, default: 140
        Diffusion steps.

    noise_scale, noise_min, noise_max: float, default: 0.1, 1e-4, 0.02
        Linear variance schedule ``noise_scale * linspace(noise_min, noise_max)``;
        the first beta is fixed at 1e-4, following the reference.

    diff_dims: tuple of int, default: (1000,)
        Hidden sizes of the denoiser MLP.

    time_emb_size: int, default: 10

    diff_dropout: float, default: 0.5
        Input dropout of the denoiser.

    diff_norm: bool, default: False
        Normalize input rows before denoising. The reference disables this.

    diff_lr: float, default: 1e-3
        Learning rate of the denoiser.

    diff_batch_size: int, default: 2048
        Users per diffusion step (rows are densified one batch at a time).

    blcc_weight: float, default: 2.0
        Weight of the bundle-level consistency term.

    rebuild_k: int, default: 1
        Bundles kept per user in the rebuilt user-bundle graph.

    num_epochs: int, default: 50
        Maximum number of training epochs.

    learning_rate: float, default: 1e-3
        Adam learning rate for the recommendation embeddings.

    batch_size: int, default: 2048
        Number of user-positive-negative bundle triples per training step.

    eval_every: int, default: 5
        Validate every this many epochs, and after the final epoch.

    patience: int, optional, default: None
        Stop after this many validation checks without improvement.
        Requires a validation set. None trains for all epochs.

    eval_k: int, default: 20
        Recall and NDCG cutoff for validation-only checkpoint selection.
        Keep a checkpoint when both metrics improve.

    device: str, optional, default: None
        PyTorch device. None selects CUDA when available, otherwise CPU.

    trainable: bool, default: True
        When False, require fitted vectors and identical catalog mappings.

    verbose: bool, default: False
        Show training progress and validation results.

    seed: int, optional, default: None
        Seed for initialization, sampling and augmentation.

    Notes
    -----
    Implementation of Li (2025); numerical objectives and settings checked against
    https://github.com/recomall/DCBR (commit 502eadf82ecfb6621d9345e18c83e1155d3786af).
    Defaults are its
    iFashion configuration. The reference hard-codes CUDA; here the device
    follows ``device``. ``noise_min``/``noise_max`` and the denoiser's
    dropout follow DiffRec, on which the reference builds.

    References
    ----------
    * Li, J. (2025). Disentangled Contrastive Bundle Recommendation with
      Conditional Diffusion. AAAI 2025.
    """

    _ann_supported = True

    def __init__(
        self,
        name="DCBR",
        emb_size=64,
        num_layers=2,
        ub_layer_weights=(0.5, 0.5, 0.0),
        ui_layer_weights=(0.3, 0.4, 0.3),
        bi_layer_weights=(0.5, 0.5, 0.0),
        view_eps=(0.05, 0.0, 0.2),
        omega=0.5,
        c_lambda=0.3,
        c_temp=0.08,
        gammas=(0.3, 0.2),
        lambda_reg=1e-6,
        diff_steps=140,
        noise_scale=0.1,
        noise_min=1e-4,
        noise_max=0.02,
        diff_dims=(1000,),
        time_emb_size=10,
        diff_dropout=0.5,
        diff_lr=1e-3,
        diff_batch_size=2048,
        blcc_weight=2.0,
        rebuild_k=1,
        num_epochs=50,
        learning_rate=1e-3,
        batch_size=2048,
        eval_every=5,
        patience=None,
        eval_k=20,
        device=None,
        trainable=True,
        verbose=False,
        seed=None,
        diff_norm=False,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.emb_size = emb_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.eval_every = eval_every
        self.patience = patience
        self.eval_k = eval_k
        self.device = device
        self.seed = seed
        self.U = None
        self.V = None
        self.num_layers = num_layers
        self.ub_layer_weights = tuple(ub_layer_weights)
        self.ui_layer_weights = tuple(ui_layer_weights)
        self.bi_layer_weights = tuple(bi_layer_weights)
        self.view_eps = tuple(view_eps)
        self.omega = omega
        self.c_lambda = c_lambda
        self.c_temp = c_temp
        self.gammas = tuple(gammas)
        self.diff_steps = diff_steps
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.diff_dims = tuple(diff_dims)
        self.time_emb_size = time_emb_size
        self.diff_dropout = diff_dropout
        self.diff_norm = diff_norm
        self.diff_lr = diff_lr
        self.diff_batch_size = diff_batch_size
        self.blcc_weight = blcc_weight
        self.rebuild_k = rebuild_k
        for key, value, minimum in (
            ("emb_size", emb_size, 1),
            ("num_epochs", num_epochs, 0),
            ("batch_size", batch_size, 1),
            ("eval_every", eval_every, 1),
            ("eval_k", eval_k, 1),
            ("num_layers", num_layers, 0),
            ("diff_steps", diff_steps, 1),
            ("rebuild_k", rebuild_k, 1),
            ("time_emb_size", time_emb_size, 2),
            ("diff_batch_size", diff_batch_size, 1),
        ):
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if patience is not None and (
            not isinstance(patience, Integral)
            or isinstance(patience, bool)
            or patience < 1
        ):
            raise ValueError("patience must be a positive integer or None")
        if not np.isfinite(lambda_reg) or lambda_reg < 0:
            raise ValueError("lambda_reg must be finite and nonnegative")
        if not self.diff_dims or any(
            not isinstance(size, Integral) or isinstance(size, bool) or size < 1
            for size in self.diff_dims
        ):
            raise ValueError("diff_dims must contain positive integers")
        for weights in (
            self.ub_layer_weights,
            self.ui_layer_weights,
            self.bi_layer_weights,
        ):
            if len(weights) != num_layers + 1:
                raise ValueError("layer weights need num_layers + 1 values")
            if (
                not np.isfinite(weights).all()
                or np.any(np.asarray(weights) < 0)
                or not sum(weights)
            ):
                raise ValueError(
                    "layer weights must be finite, nonnegative and not all zero"
                )
        if len(self.view_eps) != 3 or len(self.gammas) != 2:
            raise ValueError("view_eps needs 3 values and gammas 2 values")
        if not 0 <= omega <= 1:
            raise ValueError("omega must be in [0, 1]")
        for key, value in (
            ("learning_rate", learning_rate),
            ("c_temp", c_temp),
            ("diff_lr", diff_lr),
            ("noise_scale", noise_scale),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        coefficients = (*self.view_eps, *self.gammas, c_lambda, blcc_weight)
        if not np.isfinite(coefficients).all() or np.any(np.asarray(coefficients) < 0):
            raise ValueError(
                "noise magnitudes and loss weights must be finite and nonnegative"
            )
        if not 0 < noise_min <= noise_max or not 0 < noise_scale * noise_max < 1:
            raise ValueError(
                "noise settings require 0 < noise_min <= noise_max and noise_scale * noise_max < 1"
            )
        if not np.isfinite(noise_min) or not np.isfinite(noise_max):
            raise ValueError("noise bounds must be finite")
        if not 0 <= diff_dropout < 1:
            raise ValueError("diff_dropout must be in [0, 1)")
        if not isinstance(diff_norm, bool):
            raise ValueError("diff_norm must be a bool")

    def _build_module(self, train_set, device):
        from .dcbr import DCBRModule

        return DCBRModule(
            ui=user_item_matrix(train_set),
            ub=binary(train_set.csr_matrix),
            bi=binary(train_set.bundle_item_matrix),
            emb_size=self.emb_size,
            num_layers=self.num_layers,
            layer_weights=dict(
                zip(
                    VIEWS,
                    (
                        self.ub_layer_weights,
                        self.ui_layer_weights,
                        self.bi_layer_weights,
                    ),
                )
            ),
            view_eps=dict(zip(VIEWS, self.view_eps)),
            omega=self.omega,
            c_lambda=self.c_lambda,
            c_temp=self.c_temp,
            gammas=self.gammas,
            lambda_reg=self.lambda_reg,
            diff_steps=self.diff_steps,
            noise_scale=self.noise_scale,
            noise_min=self.noise_min,
            noise_max=self.noise_max,
            diff_dims=self.diff_dims,
            time_emb_size=self.time_emb_size,
            diff_dropout=self.diff_dropout,
            diff_norm=self.diff_norm,
            diff_lr=self.diff_lr,
            diff_batch_size=self.diff_batch_size,
            blcc_weight=self.blcc_weight,
            rebuild_k=self.rebuild_k,
            rng=self.rng,
            device=device,
        )

    def _optimizer(self, module):
        import torch

        return torch.optim.Adam(module.embedding_parameters(), lr=self.learning_rate)

    def _validate_frozen_fit(self, train_set, val_set):
        """Validate stored scoring vectors before replacing dataset references."""
        from ...data import BundleDataset

        if self.U is None or self.V is None or not self.is_fitted:
            raise ValueError("trainable=False requires fitted scoring vectors")
        if (
            self.U.ndim != 2
            or self.V.ndim != 2
            or self.U.shape != (self.num_users, self.emb_size)
            or self.V.shape != (self.num_items, self.emb_size)
        ):
            raise ValueError(
                "saved scoring vector dimensions do not match the fitted catalog"
            )
        for dataset in (train_set, val_set):
            if dataset is None:
                continue
            if not isinstance(dataset, BundleDataset):
                raise ValueError("frozen fit requires a BundleDataset")
            if (
                dataset.uid_map != self.uid_map
                or dataset.bid_map != self.bid_map
                or dataset.constituent_iid_map != self.constituent_iid_map
            ):
                raise ValueError("frozen fit requires identical ordered ID mappings")
        if (
            train_set.num_users != self.num_users
            or train_set.num_bundles != self.num_items
        ):
            raise ValueError("frozen fit requires matching training catalog dimensions")

    def _store_embeddings(self, module):
        import torch

        module.eval()
        with torch.no_grad():
            users, bundles = module.embeddings()
        self.U = users.detach().cpu().numpy().astype(np.float32)
        self.V = bundles.detach().cpu().numpy().astype(np.float32)

    def _validate(self, train_set, val_set):
        from ...eval_methods import ranking_eval
        from ...metrics import NDCG, Recall

        recall, ndcg = ranking_eval(
            model=self,
            metrics=[Recall(k=self.eval_k), NDCG(k=self.eval_k)],
            train_set=train_set,
            test_set=val_set,
        )[0]
        return recall, ndcg

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.BundleDataset`, required
            Training user-bundle feedback with bundle-item membership.

        val_set: :obj:`cornac.data.BundleDataset`, optional, default: None
            Validation feedback, used only for checkpoint selection.

        Returns
        -------
        self : object
        """
        if not self.trainable:
            self._validate_frozen_fit(train_set, val_set)
            self.train_set, self.val_set = train_set, val_set
            return self
        BundleRecommender.fit(self, train_set, val_set)
        if self.patience is not None and val_set is None:
            raise ValueError("patience requires a validation BundleDataset")
        positives = binary(train_set.csr_matrix)
        if not positives.nnz:
            raise ValueError("training requires positive user-bundle feedback")
        if np.any(np.diff(positives.indptr) == positives.shape[1]):
            raise ValueError(
                "each training user needs an unobserved bundle for BPR sampling"
            )

        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        self.rng = get_rng(self.seed)
        device = torch.device(
            self.device
            if self.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        module = self._build_module(train_set, device).to(device)
        optimizer = self._optimizer(module)
        return self._train(module, optimizer, train_set, val_set, device)

    def _train(self, module, optimizer, train_set, val_set, device):
        import torch

        positives = binary(train_set.csr_matrix)
        pair_users, pair_bundles = positives.nonzero()
        num_pairs = len(pair_users)

        self.best_epoch = None
        self.val_history = []
        best = (-np.inf, -np.inf)
        best_embeddings = None
        stale = 0

        progress = trange(self.num_epochs, desc="Training", disable=not self.verbose)
        for epoch in progress:
            module.train()
            module.epoch_start(epoch)
            order = self.rng.permutation(num_pairs)
            total = 0.0
            for start in range(0, num_pairs, self.batch_size):
                batch = order[start : start + self.batch_size]
                users = pair_users[batch]
                negatives = sample_negatives(positives, users, self.rng)
                loss, _ = module.loss(
                    torch.as_tensor(users, dtype=torch.long, device=device),
                    torch.as_tensor(
                        pair_bundles[batch], dtype=torch.long, device=device
                    ),
                    torch.as_tensor(negatives, dtype=torch.long, device=device),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            progress.set_postfix(loss=total / max(num_pairs, 1))

            last = epoch + 1 == self.num_epochs
            if val_set is None or not ((epoch + 1) % self.eval_every == 0 or last):
                continue
            self._store_embeddings(module)
            recall, ndcg = self._validate(train_set, val_set)
            self.val_history.append((epoch + 1, recall, ndcg))
            if self.verbose:
                print(
                    "epoch {}: val Recall@{k}={:.4f} NDCG@{k}={:.4f}".format(
                        epoch + 1, recall, ndcg, k=self.eval_k
                    )
                )
            if recall > best[0] and ndcg > best[1]:
                best = (recall, ndcg)
                best_embeddings = (self.U, self.V)
                self.best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
                if self.patience is not None and stale >= self.patience:
                    break

        if best_embeddings is not None:
            self.U, self.V = best_embeddings
        else:
            self._store_embeddings(module)
        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores of a user for a bundle or for all bundles.

        Parameters
        ----------
        user_idx: int, required
            The index of the user.

        item_idx: int, optional, default: None
            The index of the bundle. If None, scores of all known bundles are returned.

        Returns
        -------
        res : A scalar or a Numpy array
        """
        if self.U is None or self.V is None:
            raise ScoreException("Model must be fitted before scoring")
        if not self.knows_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)
        if item_idx is None:
            return self.V.dot(self.U[user_idx])
        if not self.knows_item(item_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d, item_id=%d)"
                % (user_idx, item_idx)
            )
        return self.V[item_idx].dot(self.U[user_idx])

    def get_vector_measure(self):
        """Score is the dot product of user and bundle vectors."""
        return MEASURE_DOT

    def get_user_vectors(self):
        """User vectors from the fused UI and BI views."""
        return self.U

    def get_item_vectors(self):
        """Bundle vectors from the fused UI and BI views."""
        return self.V
