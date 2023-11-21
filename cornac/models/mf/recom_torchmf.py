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
from tqdm.auto import trange

from ...exception import ScoreException
from ...utils.init_utils import get_rng, normal, zeros
from ..recommender import Recommender


class TorchMF(Recommender):
    """Matrix Factorization.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the latent factors.

    num_epochs: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    batch_size: int, optional, default: 256
        Batch size.

    lr: float, optional, default: 0.01
        The learning rate.

    reg: float, optional, default: 0.
        Regularization (weight_decay).

    early_stop: boolean, optional, default: False
        When True, delta loss will be checked after each iteration to stop learning earlier.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'U': user_factors, 'V': item_factors,
        'Bu': user_biases, 'Bi': item_biases}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    """

    def __init__(
        self,
        name="TorchMF",
        k=10,
        num_epochs=100,
        batch_size=256,
        lr=0.01,
        reg=0.0,
        droppout=0,
        criteria="mse",
        optimizer="sgd",
        early_stop=None,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.reg = reg
        self.droppout = droppout
        self.criteria = criteria
        self.optimizer = optimizer
        self.early_stop = early_stop
        self.seed = seed

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.u_factors = self.init_params.get("U", None)
        self.i_factors = self.init_params.get("V", None)
        self.u_biases = self.init_params.get("Bu", None)
        self.i_biases = self.init_params.get("Bi", None)

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

        import torch

        from .torchmf import MF_Pytorch, learn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        if not hasattr(self, "model"):
            self.model = MF_Pytorch(
                self.num_users,
                self.num_items,
                self.k,
                self.global_mean,
                self.droppout,
                self.init_params,
            )

        learn(
            model=self.model,
            train_set=train_set,
            n_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learn_rate=self.lr,
            reg=self.reg,
            criteria=self.criteria,
            optimizer=self.optimizer,
            device=device,
        )

        self.u_factors = self.model.u_factors.weight.detach().cpu().numpy()
        self.i_factors = self.model.i_factors.weight.detach().cpu().numpy()
        self.u_biases = self.model.u_biases.weight.detach().cpu().squeeze().numpy()
        self.i_biases = self.model.i_biases.weight.detach().cpu().squeeze().numpy()

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
            known_item_scores = np.add(self.i_biases, self.global_mean)
            if self.knows_user(user_idx):
                known_item_scores = np.add(known_item_scores, self.u_biases[user_idx])
                known_item_scores += np.dot(self.u_factors[user_idx], self.i_factors.T)
            return known_item_scores
        else:
            if not self.knows_user(user_idx) or not self.knows_item(item_idx):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_idx, item_idx))

            item_score = np.dot(self.u_factors[user_idx], self.i_factors[item_idx])
            item_score += self.global_mean + self.u_biases[user_idx] + self.i_biases[item_idx]
            return item_score
