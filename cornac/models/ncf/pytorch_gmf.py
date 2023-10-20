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


import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

from .pytorch_ncf_base import NCFBase_PyTorch
from ...exception import ScoreException


class GMF_PyTorch(NCFBase_PyTorch):
    """Generalized Matrix Factorization.

    Parameters
    ----------
    num_factors: int, optional, default: 8
        Embedding size of MF model.

    regs: float, optional, default: 0.
        Regularization for user and item embeddings.

    num_epochs: int, optional, default: 20
        Number of epochs.

    batch_size: int, optional, default: 256
        Batch size.

    num_neg: int, optional, default: 4
        Number of negative instances to pair with a positive instance.

    lr: float, optional, default: 0.001
        Learning rate.

    learner: str, optional, default: 'adam'
        Specify an optimizer: adagrad, adam, rmsprop, sgd

    early_stopping: {min_delta: float, patience: int}, optional, default: None
        If `None`, no early stopping. Meaning of the arguments: 
        
        - `min_delta`: the minimum increase in monitored value on validation set to be considered as improvement, \
           i.e. an increment of less than min_delta will count as no improvement.
        
        - `patience`: number of epochs with no improvement after which training should be stopped.

    name: string, optional, default: 'GMF'
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. \
    In Proceedings of the 26th international conference on world wide web (pp. 173-182).
    """

    def __init__(
        self,
        name="GMF-PyTorch",
        num_factors=8,
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=1e-3,
        reg=0.0,
        learner="adam",
        early_stopping=None,
        trainable=True,
        verbose=True,
        seed=None,
        use_pretrain: bool = False,
        use_NeuMF: bool = False,
        pretrained_GMF=None,
        sinkhorn=False,
        alpha=1,
        df1=None,
        df2=None,
        args=None,
    ):
        super().__init__(
            name=name,
            num_factors=num_factors,
            trainable=trainable,
            verbose=verbose,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_neg=num_neg,
            lr=lr,
            reg=reg,
            learner=learner,
            early_stopping=early_stopping,
            seed=seed,
            use_pretrain=use_pretrain,
            use_NeuMF=use_NeuMF,
            pretrained_GMF=pretrained_GMF,
        )

        self.sinkhorn = sinkhorn
        self.alpha = alpha
        self.df1 = df1
        self.df2 = df2
        self.args = args

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        super().fit(train_set, val_set)

        if self.trainable is False:
            return self

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        from .pytorch_ncf_base import GMF_torch as GMF

        self.model = GMF(
            self.num_users,
            self.num_items,
            self.num_factors,
            self.use_pretrain,
            self.use_NeuMF,
            self.pretrained_GMF,
        ).to(self.device)

        criteria = nn.MSELoss(reduction="sum")
        optimizer = self.learner(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.reg,
        )

        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            for batch_id, (batch_users, batch_items, batch_ratings) in enumerate(
                self.train_set.uir_iter(
                    self.batch_size, shuffle=True, binary=True, num_zeros=self.num_neg
                )
            ):
                batch_users = torch.from_numpy(batch_users).to(self.device)
                batch_items = torch.from_numpy(batch_items).to(self.device)
                batch_ratings = torch.tensor(batch_ratings, dtype=torch.float).to(
                    self.device
                )

                optimizer.zero_grad()
                outputs = self.model(batch_users, batch_items)
                loss = criteria(outputs, batch_ratings)
                loss.backward()
                optimizer.step()

                count += len(batch_users)
                sum_loss += loss.data.item()

                if batch_id % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))

            if self.sinkhorn:
                df1 = self.df1
                df2 = self.df2
                args = self.args
                assert df1 is not None and df2 is not None
                import geomloss

                uid_df1 = df1["user_id"].unique()
                uid_df2 = df2["user_id"].unique()
                uidx_1 = torch.tensor([train_set.uid_map[key] for key in uid_df1]).to(
                    device
                )
                uidx_2 = torch.tensor([train_set.uid_map[key] for key in uid_df2]).to(
                    device
                )
                sinkhorn_loss = geomloss.SamplesLoss(
                    loss="sinkhorn",
                    p=1,
                    blur=args.epsilon,
                    scaling=args.scaling,
                )
                l_s = self.alpha * sinkhorn_loss(
                    self.model.u_factors(uidx_1), self.model.u_factors(uidx_2)
                )
                optimizer.zero_grad()
                l_s.backward()
                optimizer.step()

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            item_ids = torch.from_numpy(np.arange(self.train_set.num_items)).to(
                self.device
            )
            user_ids = torch.tensor(user_idx).unsqueeze(0).to(self.device)

            known_item_scores = self.model.predict(user_ids, item_ids).squeeze()
            return known_item_scores.cpu().numpy()
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = self.model.predict(user_ids, item_ids).squeeze()
            return user_pred.cpu().numpy()
