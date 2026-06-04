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

import numpy as np
from tqdm.auto import trange

from cornac.models.recommender import NextItemRecommender

from ...utils import get_rng
from ..seq_utils import session_seq_iter, val_score

SUPPORTED_LOSSES = (
    "bpr",
    "bce",
    "ce",
    "bpr-max",
    "softmax",
    "cross-entropy",
    "xe_softmax",
    "top1",
)


class FPMC(NextItemRecommender):
    """Factorizing Personalized Markov Chains for next-item recommendation.

    Operates on ``(user, last_item) -> next_item`` triples. Training reuses
    the shared session iterator with ``max_len=1`` so each example contributes
    a single ``(last_item, target)`` pair; the last item only comes from the
    current session (session-based).

    Parameters
    ----------
    name: string, default: 'FPMC'

    embedding_dim: int, optional, default: 100
        Latent factor dimension.

    loss: str, optional, default: 'bpr'
        Loss function. Supported: 'bpr', 'bce', 'ce', 'bpr-max', 'softmax',
        'cross-entropy', 'xe_softmax', 'top1'.

    batch_size, learning_rate, n_sample, sample_alpha, n_epochs:
        Standard training hyperparameters.

    bpreg, elu_param: only used when ``loss="bpr-max"``.

    momentum: float, optional, default: 0.0
        Momentum for the IndexedAdagradM optimizer.

    device: str, optional, default: 'cpu'
        Set to 'cuda' for GPU support.

    model_selection: str, optional, default: 'last'
        One of 'last' or 'best'. When 'best', the model with the highest
        validation score (evaluated every ``val_eval_every`` epochs) is
        restored at the end of ``fit``.

    val_eval_every: int, optional, default: 5
    val_k: int, optional, default: 20
    val_metric: str, optional, default: 'recall'
        Cutoff and metric used for best-on-val selection. See
        :func:`cornac.models.seq_utils.val_score`.

    trainable: bool, optional, default: True
        When False, the model will not be re-trained.

    verbose: bool, optional, default: False
        When True, running logs are displayed.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    Rendle, S., Freudenthaler, C., & Schmidt-Thieme, L. (2010).
    Factorizing personalized Markov chains for next-basket recommendation. WWW.
    """

    def __init__(
        self,
        name="FPMC",
        embedding_dim=100,
        loss="bpr",
        batch_size=512,
        learning_rate=0.05,
        momentum=0.0,
        n_sample=2048,
        sample_alpha=0.5,
        n_epochs=10,
        bpreg=1.0,
        elu_param=0.5,
        device="cpu",
        trainable=True,
        verbose=False,
        seed=None,
        model_selection="last",
        val_eval_every=5,
        val_k=20,
        val_metric="recall",
    ):
        super().__init__(name, trainable=trainable, verbose=verbose)
        if loss not in SUPPORTED_LOSSES:
            raise ValueError(
                f"loss='{loss}' not supported; choose from {SUPPORTED_LOSSES}"
            )
        if model_selection not in ("last", "best"):
            raise ValueError(
                f"model_selection='{model_selection}' not supported; choose 'last' or 'best'"
            )
        self.embedding_dim = embedding_dim
        self.loss = loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.n_epochs = n_epochs
        self.bpreg = bpreg
        self.elu_param = elu_param
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)
        self.model_selection = model_selection
        self.val_eval_every = val_eval_every
        self.val_k = val_k
        self.val_metric = val_metric

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self

        import torch

        from .fpmc import FPMC_Model
        from ..seq_utils.losses import get_loss_function
        from ..seq_utils.optim import IndexedAdagradM

        torch.manual_seed(self.seed if self.seed is not None else 0)

        self.pad_idx = self.total_items
        self.model = FPMC_Model(
            user_num=self.total_users,
            item_num=self.total_items,
            factor_num=self.embedding_dim,
            pad_idx=self.pad_idx,
            device=self.device,
        ).to(self.device)

        loss_fn = get_loss_function(self.loss)
        loss_kwargs = dict(
            bpreg=self.bpreg, elu_param=self.elu_param, n_sample=self.n_sample
        )
        opt = IndexedAdagradM(
            self.model.parameters(), lr=self.learning_rate, momentum=self.momentum
        )

        best_val = -float("inf")
        best_state = None
        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for epoch_id in progress_bar:
            self.model.train()
            total_loss = 0.0
            cnt = 0
            for inc, (in_uids, hist_iids, out_iids) in enumerate(
                session_seq_iter(
                    self.train_set,
                    pad_index=self.pad_idx,
                    batch_size=self.batch_size,
                    max_len=1,
                    n_sample=self.n_sample,
                    sample_alpha=self.sample_alpha,
                    rng=self.rng,
                    shuffle=True,
                )
            ):
                if len(hist_iids) < 2:
                    continue
                in_uids_t = torch.tensor(
                    in_uids, dtype=torch.long, device=self.device, requires_grad=False
                )
                # FPMC uses just the most recent item (hist_iids shape (B, 1)).
                last_iid_t = torch.tensor(
                    hist_iids[:, -1],
                    dtype=torch.long,
                    device=self.device,
                    requires_grad=False,
                )
                out_iids_t = torch.tensor(
                    out_iids, dtype=torch.long, device=self.device, requires_grad=False
                )

                self.model.zero_grad()
                item_scores = self.model(in_uids_t, last_iid_t, out_iids_t)
                L = loss_fn(
                    item_scores,
                    out_iids=out_iids_t,
                    batch_size=len(in_uids),
                    **loss_kwargs,
                )
                L.backward()
                opt.step()

                total_loss += L.cpu().detach().numpy() * len(in_uids)
                cnt += len(in_uids)
                if inc % 10 == 0 and cnt > 0:
                    progress_bar.set_postfix(loss=(total_loss / cnt))

            if (
                self.model_selection == "best"
                and val_set is not None
                and epoch_id % self.val_eval_every == 0
            ):
                score = val_score(
                    self, self.train_set, val_set, metric=self.val_metric, k=self.val_k
                )
                if score is not None and score > best_val:
                    best_val = score
                    best_state = {
                        n: p.detach().clone()
                        for n, p in self.model.state_dict().items()
                    }

        if self.model_selection == "best" and best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def score(self, user_idx, history_items, **kwargs):
        import torch

        if len(history_items) == 0:
            return np.ones(self.total_items, dtype="float")
        last = int(history_items[-1])
        # Cap user index to known users (cold-start fallback to padding row).
        u_idx = user_idx if 0 <= user_idx < self.total_users else self.total_users
        self.model.eval()
        with torch.no_grad():
            u_t = torch.tensor([u_idx], dtype=torch.long, device=self.device)
            i_t = torch.tensor([last], dtype=torch.long, device=self.device)
            cdds = torch.arange(self.total_items, dtype=torch.long, device=self.device)
            return self.model.predict(u_t, i_t, cdds)
