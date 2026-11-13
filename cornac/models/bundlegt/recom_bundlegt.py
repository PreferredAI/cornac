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

"""Hierarchical graph-transformer catalog bundle recommendation."""

from contextlib import nullcontext

import numpy as np
from tqdm.auto import trange

from ..recommender import BundleRecommender
from ..crosscbr.recom_crosscbr import _validate_frozen_fit
from ...exception import ScoreException
from ...utils import get_rng


class BundleGT(BundleRecommender):
    """Strategy-aware Bundle Graph Transformer (BundleGT).

    Parameters
    ----------
    name: str, default: 'BundleGT'
        Model name.
    emb_size: int, default: 64
        User, item and bundle embedding dimension.
    num_epochs: int, default: 100
        Maximum training epochs.
    learning_rate: float, default: 0.001
        Adam learning rate.
    batch_size: int, default: 2048
        BPR minibatch size and maximum attention readout chunk size.
    num_ui_layers: int, default: 4
        UI LightGCN propagation layers for content embeddings.
    num_trans_layers: int, default: 3
        Hierarchical bundle-item and user-bundle attention layers.
    max_bundle_items: int, default: 70
        Maximum sorted constituent tokens per bundle; zero keeps all neighbors.
    max_user_bundles: int, default: 70
        Maximum sorted training bundle tokens per user; zero keeps all neighbors.
    gcn_norm: bool, default: False
        Normalize propagated UI layers before their sum, as in the reference.
    ub_alpha: float, default: 0.5
        User attention weight relative to the UB graph prior.
    bi_alpha: float, default: 0.5
        Bundle attention weight relative to the BI graph prior.
    dropout: float, default: 0.0
        Dropout on attention outputs; disabled for inference.
    lambda_reg: float, default: 0.00001
        Explicit L2 penalty on initial user, item and bundle embedding tables.
    early_stopping: dict, optional
        Cornac min_delta/patience settings; requires validation and restores
        vectors with best validation Recall@20.
    use_gpu: bool, default: False
        Use CUDA when requested and available.
    trainable: bool, default: True
        False reuses fitted NumPy scoring vectors.
    verbose: bool, default: False
        Display epoch training loss.
    seed: int, optional, default: 2022
        Parameter initialization, dropout and sampling seed.

    Notes
    -----
    Requires BundleDataset and independent user-item observations. Positive UB/UI
    feedback becomes binary edges. Attention uses learned Q/K, identity values,
    masked mean pooling, graph-prior residuals and hierarchical content skips.
    Q/K layers are registered in ModuleList so the optimizer trains them and
    evaluation disables their dropout. The reference implementation stores these
    layers in ordinary Python lists, outside optimizer and train/eval traversal.
    Training retains partial minibatches and checks validation Recall@20 each epoch.
    U and V contain deterministic evaluation representations, allowing prediction
    and Cornac save/load without Torch. Exact optimizer continuation is not saved.

    References
    ----------
    Wei et al. (2023). Strategy-aware Bundle Recommender System.
    SIGIR, 1198–1207. https://doi.org/10.1145/3539618.3591771

    Reference implementation: https://github.com/Xiaohao-Liu/BundleGT
    """

    def __init__(
        self,
        name="BundleGT",
        emb_size=64,
        num_epochs=100,
        learning_rate=0.001,
        batch_size=2048,
        num_ui_layers=4,
        num_trans_layers=3,
        max_bundle_items=70,
        max_user_bundles=70,
        gcn_norm=False,
        ub_alpha=0.5,
        bi_alpha=0.5,
        dropout=0.0,
        lambda_reg=1e-5,
        early_stopping=None,
        use_gpu=False,
        trainable=True,
        verbose=False,
        seed=2022,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        if emb_size < 1 or num_epochs < 1 or batch_size < 1 or num_trans_layers < 1:
            raise ValueError(
                "embedding, batch, epoch and transformer layer sizes must be positive"
            )
        if num_ui_layers < 0 or max_bundle_items < 0 or max_user_bundles < 0:
            raise ValueError("UI layers and token limits must be nonnegative")
        if learning_rate <= 0 or lambda_reg < 0:
            raise ValueError(
                "learning rate must be positive and regularization nonnegative"
            )
        if not 0 <= dropout < 1 or not 0 <= ub_alpha <= 1 or not 0 <= bi_alpha <= 1:
            raise ValueError(
                "dropout must be in [0, 1); attention weights must be in [0, 1]"
            )
        self.emb_size = emb_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_ui_layers = num_ui_layers
        self.num_trans_layers = num_trans_layers
        self.max_bundle_items = max_bundle_items
        self.max_user_bundles = max_user_bundles
        self.gcn_norm = gcn_norm
        self.ub_alpha = ub_alpha
        self.bi_alpha = bi_alpha
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.early_stopping = early_stopping
        self.use_gpu = use_gpu
        self.seed = seed
        self.U = None
        self.V = None

    def fit(self, train_set, val_set=None):
        if not self.trainable:
            _validate_frozen_fit(self, train_set, val_set)
            self.train_set = train_set
            self.val_set = val_set
            return self
        super().fit(train_set, val_set)
        if self.early_stopping is not None and val_set is None:
            raise ValueError("early_stopping requires a validation BundleDataset")
        if train_set.user_item_matrix is None:
            raise ValueError("BundleGT requires independent user_item_matrix feedback")
        ub = (train_set.csr_matrix > 0).astype(np.float32)
        ui = (train_set.user_item_matrix > 0).astype(np.float32)
        bi = (train_set.bundle_item_matrix > 0).astype(np.float32)
        if not ub.nnz or not ui.nnz:
            raise ValueError(
                "BundleGT requires positive user-bundle and user-item feedback"
            )
        if np.any(np.diff(ub.indptr) == self.num_bundles):
            raise ValueError(
                "each training user needs an unobserved bundle for BPR sampling"
            )

        import torch
        from .bundlegt import Model

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
                self.num_ui_layers,
                self.num_trans_layers,
                self.max_bundle_items,
                self.max_user_bundles,
                self.gcn_norm,
                self.ub_alpha,
                self.bi_alpha,
                self.dropout,
                self.batch_size,
                rng,
                device,
            )
            optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
            users, bundles = ub.nonzero()
            observed = [
                set(ub.indices[ub.indptr[u] : ub.indptr[u + 1]])
                for u in range(self.num_users)
            ]
            best_vectors = None
            self.loss_history = []
            epochs = trange(self.num_epochs, disable=not self.verbose, desc="BundleGT")
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
                    batch = [
                        torch.as_tensor(values, dtype=torch.long, device=device)
                        for values in (batch_users, bundles[indices], negatives)
                    ]
                    optimizer.zero_grad()
                    bpr, regularization = network(*batch)
                    loss = bpr + self.lambda_reg * regularization
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(indices)
                self.loss_history.append(epoch_loss / len(users))
                epochs.set_postfix(loss=self.loss_history[-1])
                network.eval()
                with torch.no_grad():
                    user_vectors, _, bundle_vectors = network.representations()
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
        """Score a bundle index, or all catalog bundles, using hierarchical representations."""
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
