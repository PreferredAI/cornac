# Copyright 2023 The Cornac Authors. All Rights Reserved.
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
from collections import Counter

from cornac.models.recommender import NextItemRecommender

from ...utils import get_rng


class GRU4Rec(NextItemRecommender):
    """Session-based Recommendations with Recurrent Neural Networks

    Parameters
    ----------
    name: string, default: 'GRU4Rec'
        The name of the recommender model.

    layers: list of int, optional, default: [100]
        The number of hidden units in each layer

    loss: str, optional, default: 'cross-entropy'
        Select the loss function.

    batch_size: int, optional, default: 512
        Batch size

    dropout_p_embed: float, optional, default: 0.0
        Dropout ratio for embedding layers

    dropout_p_hidden: float, optional, default: 0.0
        Dropout ratio for hidden layers

    learning_rate: float, optional, default: 0.05
        Learning rate for the optimizer

    momentum: float, optional, default: 0.0
        Momentum for adaptive learning rate

    sample_alpha: float, optional, default: 0.5
        Tradeoff factor controls the contribution of negative sample towards final loss

    n_sample: int, optional, default: 2048
        Number of negative samples

    embedding: int, optional, default: 0

    constrained_embedding: bool, optional, default: True

    n_epochs: int, optional, default: 10

    bpreg: float, optional, default: 1.0
        Regularization coefficient for 'bpr-max' loss.

    elu_param: float, optional, default: 0.5
        Elu param for 'bpr-max' loss

    logq: float, optional, default: 0,
        LogQ correction  to offset the sampling bias affecting 'cross-entropy' loss.

    device: str, optional, default: 'cpu'
        Set to 'cuda' for GPU support.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
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
    ):
        super().__init__(name, trainable=trainable, verbose=verbose)
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

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        import torch

        from .gru4rec import GRU4RecModel, IndexedAdagradM, io_iter

        item_freq = Counter(self.train_set.uir_tuple[1])
        P0 = torch.tensor(
            [item_freq[iid] for (_, iid) in self.train_set.iid_map.items()],
            dtype=torch.float32,
            device=self.device,
        ) if self.logq > 0 else None

        self.model = GRU4RecModel(
            n_items=self.total_items,
            P0=P0,
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

        opt = IndexedAdagradM(
            self.model.parameters(), self.learning_rate, self.momentum
        )

        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for _ in progress_bar:
            H = []
            for i in range(len(self.layers)):
                H.append(
                    torch.zeros(
                        (self.batch_size, self.layers[i]),
                        dtype=torch.float32,
                        requires_grad=False,
                        device=self.device,
                    )
                )
            total_loss = 0
            cnt = 0
            for inc, (in_iids, out_iids, start_mask, valid_id) in enumerate(
                io_iter(
                    s_iter=self.train_set.s_iter,
                    uir_tuple=self.train_set.uir_tuple,
                    n_sample=self.n_sample,
                    sample_alpha=self.sample_alpha,
                    rng=self.rng,
                    batch_size=self.batch_size,
                    shuffle=True,
                )
            ):
                for i in range(len(H)):
                    H[i][np.nonzero(start_mask)[0], :] = 0
                    H[i].detach_()
                    H[i] = H[i][valid_id]
                in_iids = torch.tensor(in_iids, requires_grad=False, device=self.device)
                out_iids = torch.tensor(
                    out_iids, requires_grad=False, device=self.device
                )
                self.model.zero_grad()
                R = self.model.forward(in_iids, H, out_iids, training=True)
                L = self.model.loss_function(R, out_iids, len(in_iids)) / len(in_iids)
                L.backward()
                opt.step()
                total_loss += L.cpu().detach().numpy() * len(in_iids)
                cnt += len(in_iids)
                if inc % 10 == 0:
                    progress_bar.set_postfix(loss=(total_loss / cnt))
        return self

    def score(self, user_idx, history_items, **kwargs):
        from .gru4rec import score

        if len(history_items) > 0:
            return score(self.model, self.layers, self.device, history_items)
        return np.ones(self.total_items, dtype="float")
