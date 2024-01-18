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


class DNNTSP(NextBasketRecommender):
    """Deep Neural Network for Temporal Sets Prediction (DNNTSP).

    Parameters
    ----------
    name: string, default: 'DNNTSP'
        The name of the recommender model.

    emb_dim: int, optional, default: 32
        Number of hidden factors

    loss_type: string, optional, default: "bpr"
        Loss type. Including
        "bpr": BPRLoss
        "mse": MSELoss
        "weight_mse": WeightMSELoss
        "multi_label_soft_margin": MultiLabelSoftMarginLoss

    optimizer: string, optional, default: "adam"
        Optimizer

    lr: string, optional, default: 0.001
        Learning rate

    weight_decay: float, optional, default: 0
        Weight decay for adaptive optimizer

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
    Le Yu, Leilei Sun, Bowen Du, Chuanren Liu, Hui Xiong, and Weifeng Lv. 2020.
    Predicting Temporal Sets with Deep Neural Networks.
    In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 1083–1091. https://doi.org/10.1145/3394486.3403152
    """

    def __init__(
        self,
        name="DNNTSP",
        emb_dim=32,
        loss_type="bpr",
        optimizer="adam",
        lr=0.001,
        weight_decay=0,
        n_epochs=100,
        batch_size=64,
        device="cpu",
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.emb_dim = emb_dim
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device

    def fit(self, train_set, val_set=None):
        super().fit(train_set=train_set, val_set=val_set)
        from .dnntsp import TemporalSetPrediction, learn

        self.model = TemporalSetPrediction(
            n_items=self.total_items,
            emb_dim=self.emb_dim,
            seed=self.seed,
        )

        learn(
            model=self.model,
            train_set=train_set,
            total_items=self.total_items,
            val_set=val_set,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            loss_type=self.loss_type,
            optimizer=self.optimizer,
            device=self.device,
            verbose=self.verbose,
        )

        return self

    def score(self, user_idx, history_baskets, **kwargs):
        from .dnntsp import transform_data

        self.model.eval()
        (g, nodes_feature, edges_weight, lengths, nodes, _) = transform_data(
            [history_baskets],
            item_embedding=self.model.embedding_matrix,
            total_items=self.total_items,
            device=self.device,
            is_test=True,
        )
        preds = self.model(g, nodes_feature, edges_weight, lengths, nodes)
        return preds.cpu().detach().numpy()
