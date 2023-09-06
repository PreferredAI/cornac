# Copyright 2018 The Cornac Authors. All Rights Reserved.
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

from ..recommender import Recommender
# from ...exception import ScoreException


class LightGCN(Recommender):
    def __init__(
        self,
        name="LightGCN",
        max_iter=2000,
        learning_rate=0.001,
        train_batch_size=4096,
        test_batch_size=100,
        hidden_dim=16,
        num_layers=1,
        early_stopping_patience=5,
        top_k=10,
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.early_stopping_patience = early_stopping_patience
        self.top_k = top_k
        self.seed = seed

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            from .lightgcn import Model

            self.model = Model(
                train_set.num_users,
                train_set.num_items,
                self.hidden_dim,
                self.num_layers,
                self.learning_rate,
                self.train_batch_size,
                verbose=False,
                seed=None
            )

            self.model.train(
                train_set,
                val_set,
                self.max_iter
            )

    def transform(self, test_set):
        self.user_embeddings, self.item_embeddings = self.model.predict(test_set)

    def score(self, user_idx, item_idx=None):
        # if item_idx is None:
        #     if self.train_set.is_unk_user(user_idx):
        #         raise ScoreException(
        #              "Can't make score prediction for (user_id=%d)" % user_idx
        #         )
        self.model.score(
            self.user_embeddings,
            self.item_embeddings,
            user_idx,
            item_idx
        )
