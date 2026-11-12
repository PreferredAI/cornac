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

"""Three-view fused catalog bundle recommendation."""

from contextlib import nullcontext

import numpy as np
from tqdm.auto import trange

from ..recommender import BundleRecommender
from ..crosscbr.recom_crosscbr import _validate_frozen_fit
from ...exception import ScoreException
from ...utils import get_rng


class MultiCBR(BundleRecommender):
    """MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation.

    Parameters
    ----------
    name: str, default: 'MultiCBR'
        Model name.
    emb_size: int, default: 64
        Dimension of the fused user and bundle scoring representations.
    num_epochs: int, default: 100
        Maximum training epochs.
    learning_rate: float, default: 0.001
        Adam learning rate.
    batch_size: int, default: 2048
        Minibatch size; final partial batches are retained.
    num_layers: int, default: 2
        Graph propagation layers per view.
    lambda_reg: float, default: 0.00001
        Adam weight decay on all three embedding tables.
    lambda_cl: float, default: 0.04
        Weight of the mean fused-user and positive-bundle self-contrast losses.
    temperature: float, default: 0.23
        InfoNCE temperature.
    aug_type: {'Noise', 'ED', 'MD', 'OP'}, default: 'Noise'
        Sign-aligned noise, edge dropout, message dropout, or no augmentation.
    bundle_dropout: float, default: 0.2
        UB augmentation magnitude (noise epsilon or dropout probability).
    item_dropout: float, default: 0.1
        UI propagation/pooling augmentation magnitude.
    affiliation_dropout: float, default: 0.1
        BI propagation/pooling augmentation magnitude.
    view_weights: sequence, default: (0.35, 0.15, 0.5)
        Fixed UB/UI/BI fusion coefficients; nonnegative and sum to one.
    layer_weights: sequence of three sequences, optional
        Fixed UB/UI/BI layer coefficients including the initial layer. Defaults
        to the released Youshu weights for two layers, otherwise uniform weights.
        Each sequence must have num_layers + 1 entries and sum to one.
    ed_interval: float, default: 1.0
        Edge refresh interval in epochs, with reference minibatch timing.
    early_stopping: dict, optional
        Cornac min_delta/patience settings; requires validation data and restores
        scoring vectors with the best validation Recall@20.
    use_gpu: bool, default: False
        Use CUDA when requested and available.
    trainable: bool, default: True
        False reuses previously fitted NumPy scoring vectors.
    verbose: bool, default: False
        Display training loss.
    seed: int, optional, default: 2024
        Initialization, sampling and augmentation seed.

    Notes
    -----
    Requires independent user-item feedback in BundleDataset. Positive feedback
    becomes binary graph edges. The released baseline applies self-contrast to
    one fused pass: every minibatch position, including duplicates, participates
    in the denominator. U and V store NumPy fused representations, so prediction
    and ordinary Cornac save/load do not require Torch. Training retains partial
    minibatches and checks validation Recall@20 each epoch; the reference runner
    drops partial minibatches and evaluates every five epochs.

    References
    ----------
    Ma et al. (2024). MultiCBR: Multi-view Contrastive Learning for Bundle
    Recommendation. ACM TOIS 42(4), Article 100.
    https://doi.org/10.1145/3640810

    Reference implementation: https://github.com/HappyPointer/MultiCBR
    """

    def __init__(
        self,
        name="MultiCBR",
        emb_size=64,
        num_epochs=100,
        learning_rate=0.001,
        batch_size=2048,
        num_layers=2,
        lambda_reg=1e-5,
        lambda_cl=0.04,
        temperature=0.23,
        aug_type="Noise",
        item_dropout=0.1,
        bundle_dropout=0.2,
        affiliation_dropout=0.1,
        view_weights=(0.35, 0.15, 0.5),
        layer_weights=None,
        ed_interval=1.0,
        early_stopping=None,
        use_gpu=False,
        trainable=True,
        verbose=False,
        seed=2024,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        if emb_size < 1 or num_epochs < 1 or batch_size < 1 or num_layers < 0:
            raise ValueError(
                "embedding/batch/epoch sizes must be positive and layers nonnegative"
            )
        if learning_rate <= 0 or temperature <= 0 or ed_interval <= 0:
            raise ValueError(
                "learning rate, temperature and edge interval must be positive"
            )
        if lambda_reg < 0 or lambda_cl < 0:
            raise ValueError("loss weights must be nonnegative")
        if aug_type not in ("Noise", "ED", "MD", "OP"):
            raise ValueError("aug_type must be Noise, ED, MD or OP")
        rates = (item_dropout, bundle_dropout, affiliation_dropout)
        if any(
            not np.isfinite(rate)
            or rate < 0
            or (aug_type in ("ED", "MD") and rate >= 1)
            for rate in rates
        ):
            raise ValueError(
                "augmentation magnitudes must be nonnegative; dropout must be below one"
            )
        if layer_weights is None:
            layer_weights = (
                ((0.7, 0.2, 0.1), (0.6, 0.2, 0.2), (0.6, 0.1, 0.3))
                if num_layers == 2
                else ((1 / (num_layers + 1),) * (num_layers + 1),) * 3
            )
        view_weights = np.asarray(view_weights, dtype=float)
        layer_weights = np.asarray(layer_weights, dtype=float)
        if (
            view_weights.shape != (3,)
            or not np.isfinite(view_weights).all()
            or np.any(view_weights < 0)
            or not np.isclose(view_weights.sum(), 1)
        ):
            raise ValueError(
                "view_weights must be three nonnegative coefficients summing to one"
            )
        if (
            layer_weights.shape != (3, num_layers + 1)
            or not np.isfinite(layer_weights).all()
            or np.any(layer_weights < 0)
            or not np.allclose(layer_weights.sum(axis=1), 1)
        ):
            raise ValueError(
                "layer_weights must match layers and each view must sum to one"
            )
        self.view_weights = view_weights
        self.layer_weights = layer_weights
        self.emb_size = emb_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lambda_reg = lambda_reg
        self.lambda_cl = lambda_cl
        self.temperature = temperature
        self.aug_type = aug_type
        self.item_dropout = item_dropout
        self.bundle_dropout = bundle_dropout
        self.affiliation_dropout = affiliation_dropout
        self.ed_interval = ed_interval
        self.early_stopping = early_stopping
        self.use_gpu = use_gpu
        self.seed = seed
        self.U = None
        self.V = None

    def fit(self, train_set, val_set=None):
        if not self.trainable:
            _validate_frozen_fit(self, train_set, val_set)
            return self
        super().fit(train_set, val_set)
        if self.early_stopping is not None and val_set is None:
            raise ValueError("early_stopping requires a validation BundleDataset")
        if train_set.user_item_matrix is None:
            raise ValueError("MultiCBR requires independent user_item_matrix feedback")
        ub = (train_set.csr_matrix > 0).astype(np.float32)
        ui = (train_set.user_item_matrix > 0).astype(np.float32)
        bi = (train_set.bundle_item_matrix > 0).astype(np.float32)
        if not ub.nnz or not ui.nnz:
            raise ValueError(
                "MultiCBR requires positive user-bundle and user-item feedback"
            )
        if np.any(np.diff(ub.indptr) == self.num_bundles):
            raise ValueError(
                "each training user needs an unobserved bundle for BPR sampling"
            )

        import torch
        from .multicbr import Model

        device = torch.device(
            "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        )
        devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        context = (
            torch.random.fork_rng(devices=devices)
            if self.seed is not None
            else nullcontext()
        )
        rng = get_rng(self.seed)
        with context:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            network = Model(
                ub,
                ui,
                bi,
                self.emb_size,
                self.num_layers,
                self.temperature,
                self.aug_type,
                (self.bundle_dropout, self.item_dropout, self.affiliation_dropout),
                self.view_weights,
                self.layer_weights,
                rng,
                device,
            )
            optimizer = torch.optim.Adam(
                network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.lambda_reg,
            )
            users, bundles = ub.nonzero()
            observed = [
                set(ub.indices[ub.indptr[u] : ub.indptr[u + 1]])
                for u in range(self.num_users)
            ]
            batches = (len(users) + self.batch_size - 1) // self.batch_size
            refresh = max(1, int(batches * self.ed_interval))
            steps = 0
            best_vectors = None
            self.loss_history = []
            epochs = trange(self.num_epochs, disable=not self.verbose, desc="MultiCBR")
            for _ in epochs:
                network.train()
                order = rng.permutation(len(users))
                epoch_loss = 0.0
                for start in range(0, len(order), self.batch_size):
                    indices = order[start : start + self.batch_size]
                    batch_users = users[indices]
                    negatives = rng.randint(self.num_bundles, size=len(indices))
                    for index, user in enumerate(batch_users):
                        while negatives[index] in observed[user]:
                            negatives[index] = rng.randint(self.num_bundles)
                    if self.aug_type == "ED" and (steps + 1) % refresh == 0:
                        network.refresh_graphs()
                    batch = [
                        torch.as_tensor(values, dtype=torch.long, device=device)
                        for values in (batch_users, bundles[indices], negatives)
                    ]
                    optimizer.zero_grad()
                    bpr, contrastive = network(*batch)
                    loss = bpr + self.lambda_cl * contrastive
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(indices)
                    steps += 1
                self.loss_history.append(epoch_loss / len(users))
                epochs.set_postfix(loss=self.loss_history[-1])
                network.eval()
                with torch.no_grad():
                    user_vectors, bundle_vectors = network.representations()
                    self.U = user_vectors.cpu().numpy()
                    self.V = bundle_vectors.cpu().numpy()
                if self.early_stopping is not None:
                    stopped = self.early_stop(train_set, val_set, **self.early_stopping)
                    if self.best_epoch == self.current_epoch:
                        best_vectors = (self.U.copy(), self.V.copy())
                    if stopped:
                        break
            if best_vectors is not None:
                self.U, self.V = best_vectors
        return self

    def monitor_value(self, train_set, val_set):
        if val_set is None:
            return None
        from ...eval_methods import ranking_eval
        from ...metrics import Recall

        return ranking_eval(self, [Recall(k=20)], train_set, val_set)[0][0]

    def score(self, user_idx, item_idx=None):
        """Score a bundle index, or all catalog bundles, using fused representations."""
        if (
            self.U is None
            or self.V is None
            or not self.knows_user(user_idx)
            or (item_idx is not None and not self.knows_item(item_idx))
        ):
            raise ScoreException("Cannot score an unknown user or bundle")
        if item_idx is None:
            return self.V.dot(self.U[user_idx])
        return self.V[item_idx].dot(self.U[user_idx])
