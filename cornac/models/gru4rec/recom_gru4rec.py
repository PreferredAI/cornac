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

from collections import Counter

import numpy as np
from tqdm.auto import trange

from ...utils import get_rng
from ..recommender import SequentialRecommender
from ..seq_utils import io_iter, user_io_iter

SUPPORTED_LOSSES = (
    "cross-entropy",
    "xe_softmax",
    "softmax",
    "bpr",
    "bpr-max",
    "top1",
    "bce",
    "ce",
)


class GRU4Rec(SequentialRecommender):
    """Session-based / Session-aware Recommendations with Recurrent Neural Networks.

    The model is the same GRU-based architecture in either mode; only the
    way training data is iterated differs:

    - ``mode="session-based"``: items are processed per session and the
      hidden state is reset at session boundaries. This matches the original
      GRU4Rec (Hidasi et al., 2015).
    - ``mode="session-aware"``: per-user chronological item sequences are
      concatenated across sessions and processed as a single sequence. The
      hidden state is reset at *user* boundaries instead. User IDs are not
      used in scoring but the data layout allows the RNN to capture
      cross-session dependencies.

    Parameters
    ----------
    name: string, default: 'GRU4Rec'
        The name of the recommender model.

    layers: list of int, optional, default: [100]
        The number of hidden units in each layer.

    loss: str, optional, default: 'cross-entropy'
        Loss function. Supported: 'cross-entropy', 'bpr', 'bpr-max', 'top1',
        'bce', 'ce'.

    mode: str, optional, default: 'session-based'
        Training data iteration strategy. One of 'session-based' or
        'session-aware'.

    batch_size: int, optional, default: 512
        Batch size.

    dropout_p_embed: float, optional, default: 0.0
        Dropout ratio for embedding layer.

    dropout_p_hidden: float, optional, default: 0.0
        Dropout ratio for hidden layers.

    learning_rate: float, optional, default: 0.05
        Learning rate for the optimizer.

    momentum: float, optional, default: 0.0
        Momentum for the adaptive learning rate optimizer.

    sample_alpha: float, optional, default: 0.5
        Tradeoff factor controls the contribution of negative samples
        towards the final loss (popularity-based sampling exponent).

    n_sample: int, optional, default: 2048
        Number of additional shared negative samples per mini-batch.

    embedding: int, optional, default: 0
        Size of the separate input embedding. ``0`` means no separate
        embedding; use ``"layersize"`` to set it to ``layers[0]``.

    constrained_embedding: bool, optional, default: True
        Whether input and output item embeddings are tied.

    n_epochs: int, optional, default: 10

    bpreg: float, optional, default: 1.0
        Regularization coefficient for 'bpr-max' loss.

    elu_param: float, optional, default: 0.5
        ELU parameter for 'bpr-max' loss.

    logq: float, optional, default: 0.0
        LogQ correction strength to offset sampling bias for the
        cross-entropy loss.

    device: str, optional, default: 'cpu'
        Set to 'cuda' for GPU support.

    trainable: bool, optional, default: True
        When False, the model will not be re-trained.

    verbose: bool, optional, default: False
        When True, running logs are displayed.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    Hidasi, B., Karatzoglou, A., Baltrunas, L., & Tikk, D. (2015).
    Session-based recommendations with recurrent neural networks.
    arXiv preprint arXiv:1511.06939.
    """

    def __init__(
        self,
        name="GRU4Rec",
        layers=[100],
        loss="cross-entropy",
        mode="session-based",
        batch_size=512,
        dropout_p_embed=0.0,
        dropout_p_hidden=0.0,
        learning_rate=0.05,
        momentum=0.0,
        sample_alpha=0.5,
        n_sample=2048,
        embedding=0,
        constrained_embedding=True,
        n_epochs=10,
        bpreg=1.0,
        elu_param=0.5,
        logq=0.0,
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
        self.layers = layers
        self.loss = loss
        self.batch_size = batch_size
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.sample_alpha = sample_alpha
        self.n_sample = n_sample
        self.embedding = self.layers[0] if embedding == "layersize" else embedding
        self.constrained_embedding = constrained_embedding
        self.n_epochs = n_epochs
        self.bpreg = bpreg
        self.elu_param = elu_param
        self.logq = logq
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

    def _build_loss_kwargs(self):
        return dict(
            P0=self.P0,
            logq=self.logq,
            sample_alpha=self.sample_alpha,
            batch_size=None,  # filled per-batch
            bpreg=self.bpreg,
            elu_param=self.elu_param,
            n_sample=self.n_sample,
        )

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self

        import torch

        from .gru4rec import GRU4RecModel
        from ..seq_utils.losses import get_loss_function
        from ..seq_utils.optim import IndexedAdagradM

        item_freq = Counter(self.train_set.uir_tuple[1])
        self.P0 = (
            torch.tensor(
                [item_freq[iid] for (_, iid) in self.train_set.iid_map.items()],
                dtype=torch.float32,
                device=self.device,
            )
            if self.logq > 0
            else None
        )

        self.model = GRU4RecModel(
            n_items=self.total_items,
            P0=self.P0,
            layers=self.layers,
            dropout_p_embed=self.dropout_p_embed,
            dropout_p_hidden=self.dropout_p_hidden,
            embedding=self.embedding,
            constrained_embedding=self.constrained_embedding,
            logq=self.logq,
            sample_alpha=self.sample_alpha,
            bpreg=self.bpreg,
            elu_param=self.elu_param,
            loss=self.loss,
        ).to(self.device)
        self.model._reset_weights_to_compatibility_mode()

        loss_fn = get_loss_function(self.loss)
        loss_kwargs = self._build_loss_kwargs()

        opt = IndexedAdagradM(
            self.model.parameters(), self.learning_rate, self.momentum
        )

        best_val = -float("inf")
        best_state = None
        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for epoch_id in progress_bar:
            H = [
                torch.zeros(
                    (self.batch_size, self.layers[i]),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=self.device,
                )
                for i in range(len(self.layers))
            ]
            total_loss = 0.0
            cnt = 0
            data_iter = self._build_data_iter()
            for inc, batch in enumerate(data_iter):
                # Unified 5-tuple from io_iter / user_io_iter:
                #   (in_uids, in_iids, out_iids, start_mask, valid_id)
                _, in_iids, out_iids, start_mask, valid_id = batch
                for i in range(len(H)):
                    H[i][np.nonzero(start_mask)[0], :] = 0
                    H[i].detach_()
                    H[i] = H[i][valid_id]
                in_iids_t = torch.tensor(
                    in_iids, dtype=torch.long, requires_grad=False, device=self.device
                )
                out_iids_t = torch.tensor(
                    out_iids, dtype=torch.long, requires_grad=False, device=self.device
                )
                self.model.zero_grad()
                R = self.model.forward(in_iids_t, H, out_iids_t, training=True)
                loss_kwargs["batch_size"] = len(in_iids)
                loss_kwargs["out_iids"] = out_iids_t
                L = loss_fn(R, **loss_kwargs)
                L.backward()
                opt.step()
                total_loss += L.cpu().detach().numpy() * len(in_iids)
                cnt += len(in_iids)
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

    def _build_data_iter(self):
        if self.mode == "session-aware":
            return user_io_iter(
                train_set=self.train_set,
                n_sample=self.n_sample,
                sample_alpha=self.sample_alpha,
                rng=self.rng,
                batch_size=self.batch_size,
                shuffle=True,
            )
        return io_iter(
            s_iter=self.train_set.s_iter,
            uir_tuple=self.train_set.uir_tuple,
            n_sample=self.n_sample,
            sample_alpha=self.sample_alpha,
            rng=self.rng,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def score(self, user_idx, history_items, **kwargs):
        from .gru4rec import score as _score

        flat_history = self._flatten_history(history_items)
        if len(flat_history) == 0:
            return np.ones(self.total_items, dtype="float")
        scores = _score(self.model, self.layers, self.device, flat_history)
        if scores is None:
            return np.ones(self.total_items, dtype="float")
        # The output embedding has +1 padding row; trim if present
        if scores.shape[-1] == self.total_items + 1:
            scores = scores[: self.total_items]
        return scores
