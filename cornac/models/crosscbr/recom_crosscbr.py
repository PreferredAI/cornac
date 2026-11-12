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

"""Cross-view contrastive catalog bundle recommendation."""

from contextlib import nullcontext

import numpy as np
from tqdm.auto import trange

from ..recommender import BundleRecommender
from ...exception import ScoreException
from ...utils import get_rng


def _validate_frozen_fit(model, train_set, val_set=None):
    """Check saved scoring vectors against the ordered bundle catalog."""
    from ...data import BundleDataset

    if model.U is None or model.V is None or not model.is_fitted:
        raise ValueError("trainable=False requires fitted scoring vectors")
    if (
        model.U.ndim != 2
        or model.V.ndim != 2
        or model.U.shape[0] != model.num_users
        or model.V.shape[0] != model.num_items
        or model.U.shape[1] != model.V.shape[1]
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
            dataset.uid_map != model.uid_map
            or dataset.bid_map != model.bid_map
            or dataset.constituent_iid_map != model.constituent_iid_map
        ):
            raise ValueError("frozen fit requires identical ordered ID mappings")
    if (
        train_set.num_users != model.num_users
        or train_set.num_bundles != model.num_bundles
    ):
        raise ValueError("frozen fit requires matching training catalog dimensions")


class CrossCBR(BundleRecommender):
    """CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation.

    Parameters
    ----------
    name: str, default: 'CrossCBR'
        Model name.
    emb_size: int, default: 64
        Embedding dimension for each view; scoring vectors have twice this size.
    num_epochs: int, default: 100
        Maximum training epochs.
    learning_rate: float, default: 0.001
        Adam learning rate.
    batch_size: int, default: 2048
        User-bundle minibatch size, including the last partial batch.
    num_layers: int, default: 1
        Propagation layers in each view.
    lambda_reg: float, default: 0.0001
        Adam weight decay over the three initial embedding tables.
    lambda_cl: float, default: 0.04
        Weight of the mean user and positive-bundle cross-view contrastive losses.
    temperature: float, default: 0.25
        InfoNCE temperature.
    aug_type: {'ED', 'MD', 'OP'}, default: 'ED'
        Edge dropout, message dropout, or original preservation.
    item_dropout: float, default: 0.2
        Dropout rate for the user-item view.
    bundle_dropout: float, default: 0.2
        Dropout rate for the user-bundle view.
    aggregation_dropout: float, default: 0.2
        Dropout rate for bundle-item pooling. All rates are ignored for OP.
    ed_interval: float, default: 1.0
        Edge refresh interval measured in epochs, converted to at least one
        minibatch. As in the reference, refresh occurs before every interval's
        last batch; the initial dropped graphs are used before that refresh.
    early_stopping: dict, optional
        Cornac ``min_delta`` and ``patience`` settings. Requires validation data;
        monitors Recall@20 each epoch and restores the best scoring vectors.
    use_gpu: bool, default: False
        Use CUDA when requested and available.
    trainable: bool, default: True
        Whether to train; False requires previously fitted scoring vectors.
    verbose: bool, default: False
        Display epoch loss.
    seed: int, optional, default: 2022
        Initialization, dropout and sampling seed. CPU training is reproducible.

    Notes
    -----
    Requires BundleDataset with independently observed user_item_matrix. Positive
    entries are binarized; zero/negative feedback is ignored. Users with every
    catalog bundle observed are rejected because no BPR negative exists.
    ``U`` and ``V`` contain concatenated item/bundle-view NumPy representations.
    Loading and prediction do not require Torch. Training retains partial
    minibatches and checks validation Recall@20 each epoch.

    References
    ----------
    Ma, Y., He, Y., Zhang, A., Wang, X., & Chua, T.-S. (2022). CrossCBR:
    Cross-view Contrastive Learning for Bundle Recommendation. KDD, 1233–1241.
    https://doi.org/10.1145/3534678.3539229

    Reference implementation: https://github.com/mysbupt/CrossCBR
    """

    def __init__(
        self,
        name="CrossCBR",
        emb_size=64,
        num_epochs=100,
        learning_rate=0.001,
        batch_size=2048,
        num_layers=1,
        lambda_reg=1e-4,
        lambda_cl=0.04,
        temperature=0.25,
        aug_type="ED",
        item_dropout=0.2,
        bundle_dropout=0.2,
        aggregation_dropout=0.2,
        ed_interval=1.0,
        early_stopping=None,
        use_gpu=False,
        trainable=True,
        verbose=False,
        seed=2022,
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
        if aug_type not in ("ED", "MD", "OP"):
            raise ValueError("aug_type must be ED, MD or OP")
        if any(
            not 0 <= rate < 1
            for rate in (item_dropout, bundle_dropout, aggregation_dropout)
        ):
            raise ValueError("dropout rates must be in [0, 1)")
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
        self.aggregation_dropout = aggregation_dropout
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
            raise ValueError("CrossCBR requires independent user_item_matrix feedback")
        ub = (train_set.csr_matrix > 0).astype(np.float32)
        ui = (train_set.user_item_matrix > 0).astype(np.float32)
        bi = (train_set.bundle_item_matrix > 0).astype(np.float32)
        if not ub.nnz or not ui.nnz:
            raise ValueError(
                "CrossCBR requires positive user-bundle and user-item feedback"
            )
        if np.any(np.diff(ub.indptr) == self.num_bundles):
            raise ValueError(
                "each training user needs an unobserved bundle for BPR sampling"
            )

        import torch
        from .crosscbr import Model

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
                (self.item_dropout, self.bundle_dropout, self.aggregation_dropout),
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
            epochs = trange(self.num_epochs, disable=not self.verbose, desc="CrossCBR")
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
                    user_views, bundle_views = network.representations()
                    self.U = torch.cat(user_views, dim=1).cpu().numpy()
                    self.V = torch.cat(bundle_views, dim=1).cpu().numpy()
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
        """Score a bundle index, or all catalog bundles, using both learned views."""
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
