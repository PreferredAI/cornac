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

from ...utils import get_rng
from ..recommender import SequentialRecommender
from ..seq_utils import session_seq_iter, user_seq_iter

SUPPORTED_LOSSES = (
    "bce",
    "ce",
    "bpr",
    "bpr-max",
    "softmax",
    "cross-entropy",
    "xe_softmax",
    "top1",
)


class GPT2Rec(SequentialRecommender):
    """GPT2-based causal transformer for sequential recommendation.

    Parameters mirror :class:`cornac.models.SASRec`.
    """

    def __init__(
        self,
        name="GPT2Rec",
        embedding_dim=100,
        mode="session-aware",
        loss="ce",
        batch_size=512,
        learning_rate=0.001,
        n_sample=2048,
        sample_alpha=0.5,
        n_epochs=10,
        max_len=50,
        num_blocks=2,
        num_heads=1,
        dropout=0.2,
        l2_reg=0.0,
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
        super().__init__(name, mode=mode, trainable=trainable, verbose=verbose)
        if loss not in SUPPORTED_LOSSES:
            raise ValueError(
                f"loss='{loss}' not supported; choose from {SUPPORTED_LOSSES}"
            )
        self.embedding_dim = embedding_dim
        self.loss = loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.n_epochs = n_epochs
        self.max_len = max_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.bpreg = bpreg
        self.elu_param = elu_param
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)
        if model_selection not in ("last", "best"):
            raise ValueError(
                f"model_selection='{model_selection}' not supported; choose 'last' or 'best'"
            )
        self.model_selection = model_selection
        self.val_eval_every = val_eval_every
        self.val_k = val_k
        self.val_metric = val_metric

    def _build_data_iter(self, pad_index):
        kwargs = dict(
            train_set=self.train_set,
            pad_index=pad_index,
            batch_size=self.batch_size,
            max_len=self.max_len,
            n_sample=self.n_sample,
            sample_alpha=self.sample_alpha,
            rng=self.rng,
            shuffle=True,
        )
        if self.mode == "session-aware":
            return user_seq_iter(**kwargs)
        return session_seq_iter(**kwargs)

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self

        import torch

        from .gpt2rec import GPT2RecModel
        from ..seq_utils.losses import get_loss_function

        torch.manual_seed(self.seed if self.seed is not None else 0)

        self.pad_idx = self.total_items
        self.model = GPT2RecModel(
            item_num=self.total_items,
            embedding_dim=self.embedding_dim,
            maxlen=self.max_len,
            n_layers=self.num_blocks,
            n_heads=self.num_heads,
            dropout=self.dropout,
            pad_idx=self.pad_idx,
            device=self.device,
        )

        loss_fn = get_loss_function(self.loss)
        loss_kwargs = dict(
            bpreg=self.bpreg, elu_param=self.elu_param, n_sample=self.n_sample
        )

        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98)
        )

        best_val = -float("inf")
        best_state = None
        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for epoch_id in progress_bar:
            self.model.train()
            total_loss = 0.0
            cnt = 0
            for inc, (in_uids, hist_iids, out_iids) in enumerate(
                self._build_data_iter(self.pad_idx)
            ):
                if len(hist_iids) < 2:
                    continue
                hist_iids_t = torch.tensor(
                    hist_iids, dtype=torch.long, device=self.device, requires_grad=False
                )
                out_iids_t = torch.tensor(
                    out_iids, dtype=torch.long, device=self.device, requires_grad=False
                )

                self.model.zero_grad()
                item_scores = self.model(
                    None, hist_iids_t, out_iids_t, return_hidden=False
                )

                L = loss_fn(
                    item_scores,
                    out_iids=out_iids_t,
                    batch_size=len(hist_iids),
                    **loss_kwargs,
                )
                if self.l2_reg > 0:
                    for p in self.model.parameters():
                        L = L + self.l2_reg * torch.norm(p)

                L.backward()
                opt.step()

                total_loss += L.cpu().detach().numpy() * len(hist_iids)
                cnt += len(hist_iids)
                if inc % 10 == 0 and cnt > 0:
                    progress_bar.set_postfix(loss=(total_loss / cnt))

            if (
                self.model_selection == "best"
                and val_set is not None
                and epoch_id % self.val_eval_every == 0
            ):
                val_score = self._val_score(
                    val_set, metric=self.val_metric, k=self.val_k
                )
                if val_score is not None and val_score > best_val:
                    best_val = val_score
                    best_state = {
                        n: p.detach().clone() for n, p in self.model.state_dict().items()
                    }
                if self.verbose:
                    progress_bar.set_postfix(
                        loss=(total_loss / max(cnt, 1)),
                        val_recall=val_score,
                        best=best_val,
                    )

        if self.model_selection == "best" and best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def score(self, user_idx, history_items, **kwargs):
        import torch

        flat = self._flatten_history(history_items)
        if len(flat) == 0:
            return np.ones(self.total_items, dtype="float")
        log_seq = [self.pad_idx] * (self.max_len - len(flat)) + list(flat)
        log_seq = log_seq[-self.max_len :]
        log_seq_t = torch.tensor([log_seq], dtype=torch.long, device=self.device)
        self.model.eval()
        return self.model.predict(user_idx, log_seq_t)
