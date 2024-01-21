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

from ..recommender import NextBasketRecommender


class DREAM(NextBasketRecommender):
    """Dynamic Recurrent Basket Model (DREAM)

    Parameters
    ----------
    name: string, default: 'DREAM'
        The name of the recommender model.

    emb_size: int, optional, default: 32
        Embedding size

    emb_type: str, default: 'mean'
        Embedding type. Including 'mean', 'max', 'sum'

    hidden_size: int, optional, default: 32
        Hidden size

    dropout: float, optional, default: 0.1
        Dropout ratio

    loss_mode: int, optional, default: 0
        Loss mode. Including 0 and 1

    loss_uplift: int, optional, default: 100

    attention: int, optional, default: 0
        Attention

    max_seq_length: int, optional, default: None
        Max sequence length.
        If None, it is the maximum number of baskets in training sequences

    lr: string, optional, default: 0.001
        Learning rate of Adam optimizer

    n_epochs: int, optional, default: 100
        Number of epochs

    batch_size: int, optional, default: 64
        Batch size

    device: string, optional, default: "cpu"
        Device for learning and evaluation. Using cpu as default.
        Use "cuda:0" for using gpu.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    seed: int, optional, default: None
        Random seed

    References
    ----------
    Feng Yu, Qiang Liu, Shu Wu, Liang Wang, and Tieniu Tan. 2016.
    A Dynamic Recurrent Model for Next Basket Recommendation.
    In Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval (SIGIR '16).
    Association for Computing Machinery, New York, NY, USA, 729–732.
    https://doi.org/10.1145/2911451.2914683

    """

    def __init__(
        self,
        name="DREAM",
        emb_size=32,
        emb_type="mean",
        hidden_size=32,
        dropout=0.1,
        loss_mode=0,
        loss_uplift=100,
        attention=0,
        max_seq_length=None,
        lr=0.001,
        weight_decay=0,
        n_epochs=100,
        batch_size=32,
        device="cpu",
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.emb_size = emb_size
        self.emb_type = emb_type
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.loss_mode = loss_mode
        self.loss_uplift = loss_uplift
        self.attention = attention
        self.max_seq_length = max_seq_length
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

    def fit(self, train_set, val_set=None):
        super().fit(train_set=train_set, val_set=val_set)
        from .dream import DREAM, learn

        # max sequence length
        self.max_seq_length = (
            max([len(bids) for bids in train_set.user_basket_data.values()])
            if self.max_seq_length is None
            else self.max_seq_length
        )
        self.model = DREAM(
            n_items=self.total_items,
            emb_size=self.emb_size,
            emb_type=self.emb_type,
            hidden_size=self.hidden_size,
            dropout_prob=self.dropout,
            max_seq_length=self.max_seq_length,
            loss_mode=self.loss_mode,
            loss_uplift=self.loss_uplift,
            attention=self.attention,
            device=self.device,
            seed=self.seed,
        ).to(self.device)

        learn(
            self.model,
            train_set=train_set,
            val_set=val_set,
            max_seq_length=self.max_seq_length,
            lr=self.lr,
            weight_decay=self.weight_decay,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        return self

    def score(self, user_idx, history_baskets, **kwargs):
        self.model.eval()
        preds = self.model([history_baskets[-self.max_seq_length :]])
        return preds.squeeze().cpu().detach().numpy()
