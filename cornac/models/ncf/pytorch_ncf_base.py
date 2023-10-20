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

from ..recommender import Recommender
from ...utils import get_rng


class NCFBase_PyTorch(Recommender):
    """
    Parameters
    ----------
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

    name: string, optional, default: 'Torch-NCF'
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.
    """

    def __init__(
        self,
        name="NCF",
        num_factors=8,
        layers=None,
        act_fn="relu",
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
        pretrained_MLP=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.num_factors = num_factors
        self.layers = layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.lr = lr
        self.reg = reg
        self.early_stopping = early_stopping
        self.seed = seed
        self.rng = get_rng(seed)
        self.use_pretrain = use_pretrain
        self.use_NeuMF = use_NeuMF
        self.pretrained_GMF = pretrained_GMF
        self.pretrained_MLP = pretrained_MLP

        optimizer = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }
        self.learner = optimizer[learner.lower()]

        activation_functions = {
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "relu": nn.ReLU(),
            "relu6": nn.ReLU6(),
            "leakyrelu": nn.LeakyReLU(),
        }
        self.act_fn = activation_functions[act_fn.lower()]

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
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            if not hasattr(self, "user_embedding"):
                self.num_users = self.train_set.num_users
                self.num_items = self.train_set.num_items

        return self

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
        raise NotImplementedError("The algorithm is not able to make score prediction!")

    def monitor_value(self):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.

        Returns
        -------
        res : float
            Monitored value on validation set.
            Return `None` if `val_set` is `None`.
        """
        if self.val_set is None:
            return None

        from ...metrics import Recall
        from ...eval_methods import ranking_eval

        recall_20 = ranking_eval(
            model=self,
            metrics=[Recall(k=20)],
            train_set=self.train_set,
            test_set=self.val_set,
        )[0][0]

        return recall_20


class GMF_torch(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 8,
        use_pretrain: bool = False,
        use_NeuMF: bool = False,
        pretrained_GMF=None,
    ):
        super(GMF_torch, self).__init__()

        self.pretrained_GMF = pretrained_GMF
        self.num_users = num_users
        self.num_items = num_items
        self.use_pretrain = use_pretrain
        self.use_NeuMF = use_NeuMF

        self.pretrained_GMF = pretrained_GMF

        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)

        self.predict_layer = nn.Linear(num_factors, 1)
        self.Sigmoid = nn.Sigmoid()

        if use_pretrain:
            self._load_pretrained_model()
        else:
            self._init_weight()

    def _init_weight(self):
        if not self.use_pretrain:
            nn.init.normal_(self.user_embedding.weight, std=1e-2)
            nn.init.normal_(self.item_embedding.weight, std=1e-2)
        if not self.use_NeuMF:
            nn.init.normal_(self.predict_layer.weight, std=1e-2)

    def _load_pretrained_model(self):
        self.user_embedding.weight.data.copy_(self.pretrained_GMF.user_embedding.weight)
        self.item_embedding.weight.data.copy_(self.pretrained_GMF.item_embedding.weight)

    def forward(self, users, items):
        embedding_elementwise = self.user_embedding(users) * self.item_embedding(items)
        if not self.use_NeuMF:
            output = self.predict_layer(embedding_elementwise)
            output = self.Sigmoid(output)
            output = output.view(-1)
        else:
            output = embedding_elementwise

        return output

    def predict(self, users, items):
        with torch.no_grad():
            preds = (self.user_embedding(users) * self.item_embedding(items)).sum(
                dim=1, keepdim=True
            )
        return preds.squeeze()


class MLP_torch(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 8,
        layers=None,
        act_fn=nn.ReLU(),
        use_pretrain: bool = False,
        use_NeuMF: bool = False,
        pretrained_MLP=None,
    ):
        super(MLP_torch, self).__init__()

        if layers is None:
            layers = [64, 32, 16, 8]

        self.pretrained_MLP = pretrained_MLP
        self.num_users = num_users
        self.num_items = num_items
        self.use_pretrain = use_pretrain
        self.user_embedding = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(num_items, layers[0] // 2)
        self.use_NeuMF = use_NeuMF
        MLP_layers = []

        for idx, factor in enumerate(layers[:-1]):
            # ith MLP layer (layer[i],layer[i]//2) -> #(i+1)th MLP layer (layer[i+1],layer[i+1]//2)
            # ex) (64,32) -> (32,16) -> (16,8)
            # MLP_layers.append(nn.Linear(factor, factor // 2))

            MLP_layers.append(nn.Linear(factor, layers[idx + 1]))
            MLP_layers.append(act_fn)

        # unpacking layers in to torch.nn.Sequential
        self.MLP_model = nn.Sequential(*MLP_layers)

        self.predict_layer = nn.Linear(num_factors, 1)
        self.Sigmoid = nn.Sigmoid()

        if self.use_pretrain:
            self._load_pretrained_model()
        else:
            self._init_weight()

    def _init_weight(self):
        if not self.use_pretrain:
            nn.init.normal_(self.user_embedding.weight, std=1e-2)
            nn.init.normal_(self.item_embedding.weight, std=1e-2)
            for layer in self.MLP_model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
        if not self.use_NeuMF:
            nn.init.normal_(self.predict_layer.weight, std=1e-2)

    def _load_pretrained_model(self):
        self.user_embedding.weight.data.copy_(self.pretrained_MLP.user_embedding.weight)
        self.item_embedding.weight.data.copy_(self.pretrained_MLP.item_embedding.weight)
        for layer, pretrained_layer in zip(
            self.MLP_model, self.pretrained_MLP.MLP_model
        ):
            if isinstance(layer, nn.Linear) and isinstance(pretrained_layer, nn.Linear):
                layer.weight.data.copy_(pretrained_layer.weight)
                layer.bias.data.copy_(pretrained_layer.bias)

    def forward(self, user, item):
        embed_user = self.user_embedding(user)
        embed_item = self.item_embedding(item)
        embed_input = torch.cat((embed_user, embed_item), dim=-1)
        output = self.MLP_model(embed_input)

        if not self.use_NeuMF:
            output = self.predict_layer(output)
            output = self.Sigmoid(output)
            output = output.view(-1)

        return output

    def predict(self, users, items):
        with torch.no_grad():
            embed_user = self.user_embedding(users)
            if len(users) == 1:
                # replicate user embedding to len(items)
                embed_user = embed_user.repeat(len(items), 1)
            embed_item = self.item_embedding(items)
            embed_input = torch.cat((embed_user, embed_item), dim=-1)
            output = self.MLP_model(embed_input)

            output = self.predict_layer(output)
            output = self.Sigmoid(output)
            output = output.view(-1)
        return output

    def __call__(self, *args):
        return self.forward(*args)


class NeuMF_torch(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 8,
        layers=None,  # layer for MLP
        act_fn=nn.ReLU(),
        use_pretrain: bool = False,
        pretrained_GMF=None,
        pretrained_MLP=None,
    ):
        super(NeuMF_torch, self).__init__()

        self.use_pretrain = use_pretrain
        self.pretrained_GMF = pretrained_GMF
        self.pretrained_MLP = pretrained_MLP

        # layer for MLP
        if layers is None:
            layers = [64, 32, 16, 8]

        self.predict_layer = nn.Linear(num_factors * 2, 1)
        self.Sigmoid = nn.Sigmoid()

        self.GMF = GMF_torch(
            num_users,
            num_items,
            num_factors,
            use_pretrain=use_pretrain,
            use_NeuMF=True,
            pretrained_GMF=self.pretrained_GMF,
        )
        self.MLP = MLP_torch(
            num_users=num_users,
            num_items=num_items,
            num_factors=num_factors,
            layers=layers,
            act_fn=act_fn,
            use_pretrain=use_pretrain,
            use_NeuMF=True,
            pretrained_MLP=self.pretrained_MLP,
        )

        if self.use_pretrain:
            self._load_pretrain_model()

        if not self.use_pretrain:
            nn.init.normal_(self.predict_layer.weight, std=1e-2)

    def _load_pretrain_model(self):
        predict_weight = torch.cat(
            [
                self.pretrained_GMF.predict_layer.weight,
                self.pretrained_MLP.predict_layer.weight,
            ],
            dim=1,
        )
        predict_bias = (
            self.pretrained_GMF.predict_layer.bias
            + self.pretrained_MLP.predict_layer.bias
        )
        self.predict_layer.weight.data.copy_(0.5 * predict_weight)
        self.predict_layer.bias.data.copy_(0.5 * predict_bias)

    def forward(self, user, item):
        before_last_layer_output = torch.cat(
            (self.GMF(user, item), self.MLP(user, item)), dim=-1
        )
        output = self.predict_layer(before_last_layer_output)
        output = self.Sigmoid(output)
        return output.view(-1)

    def predict(self, users, items):
        with torch.no_grad():
            if len(users) == 1:
                # replicate user embedding to len(items)
                users = users.repeat(len(items))
            # breakpoint()
            before_last_layer_output = torch.cat(
                (self.GMF(users, items), self.MLP(users, items)), dim=-1
            )
            preds = self.predict_layer(before_last_layer_output)
            preds = self.Sigmoid(preds)
        return preds.view(-1)
