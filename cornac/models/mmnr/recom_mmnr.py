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


class MMNR(NextBasketRecommender):
    """Multi-view Multi-aspect Neural Recommendation.

    Parameters
    ----------
    name: string, default: 'MMNR'
        The name of the recommender model.

    References
    ----------
    Zhiying Deng, Jianjun Li, Zhiqiang Guo, Wei Liu, Li Zou, and Guohui Li. 2023.
    Multi-view Multi-aspect Neural Networks for Next-basket Recommendation.
    In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23).
    Association for Computing Machinery, New York, NY, USA, 1283–1292. https://doi.org/10.1145/3539618.3591738
    """

    def __init__(
        self,
        name="MMNR",
        emb_dim=32,
        n_aspects=11,
        ctx=3,
        d1=5,
        d2=5,
        decay=0.6,
        lr=1e-2,
        l2=1e-3,
        optimizer="adam",
        batch_size=100,
        n_epochs=20,
        m=1,
        n=0.002,
        device="cpu",
        init_params=None,
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.emb_dim = emb_dim
        self.n_aspects = n_aspects
        self.seed = seed
        self.ctx = ctx
        self.d1 = d1
        self.d2 = d2
        self.optimizer = optimizer
        self.lr = lr
        self.l2 = l2
        self.m = m
        self.n = n
        self.decay = decay
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.init_params = init_params if init_params is not None else {}

    def fit(self, train_set, val_set=None):
        super().fit(train_set=train_set, val_set=val_set)
        from .mmnr import Model, build_history_matrix, learn

        self.model = Model(
            self.total_items,
            emb_dim=self.emb_dim,
            n_aspects=self.n_aspects,
            padding_idx=self.total_items,
            ctx=self.ctx,
            d1=self.d1,
            d2=self.d2,
        )
        learn(
            model=self.model,
            train_set=train_set,
            total_users=self.total_users,
            total_items=self.total_items,
            val_set=val_set,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            l2=self.l2,
            m=self.m,
            n=self.n,
            decay=self.decay,
            optimizer=self.optimizer,
            device=self.device,
            verbose=self.verbose,
        )

        self.user_history_matrix = self.init_params.get("user_history_matrix", None)
        self.item_history_matrix = self.init_params.get("item_history_matrix", None)
        if self.user_history_matrix is None or self.item_history_matrix is None:
            print(
                "Constructing test history matrices from train_set and val_set as they are not provided."
            )
            self.user_history_matrix, self.item_history_matrix = build_history_matrix(
                train_set=train_set,
                val_set=val_set,
                test_set=None,
                total_users=self.total_users,
                total_items=self.total_items,
                mode="test",
            )
        return self

    def score(self, user_idx, history_baskets, **kwargs):
        from .mmnr import score

        item_scores = score(
            self.model,
            self.user_history_matrix,
            self.item_history_matrix,
            self.total_items,
            user_idx,
            history_baskets,
            self.decay,
            self.device,
        )
        return item_scores
