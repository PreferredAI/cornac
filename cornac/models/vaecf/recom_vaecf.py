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

from ..recommender import Recommender
from ...exception import ScoreException


class VAECF(Recommender):
    """Variational Autoencoder for Collaborative Filtering.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the stochastic user factors ``z''.

    h: int, optional, default: 20
       The dimension of the deterministic hidden layer.

    n_epochs: int, optional, default: 100
        The number of epochs for SGD.

    batch_size: int, optional, default: 100
        The batch size.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD_RMSProp.

    beta: float, optional, default: 1.
        The weight of the KL term as in beta-VAE.
		
    name: string, optional, default: 'VAECF'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.
		
    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    References
    ----------
    * Liang, Dawen, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. "Variational autoencoders for collaborative filtering." \
	In Proceedings of the 2018 World Wide Web Conference on World Wide Web, pp. 689-698.
    """

    def __init__(self, name="VAECF", k=10, h=20, n_epochs=100, batch_size=100, learning_rate=0.001, beta=1.,
                 trainable=True, verbose=False, seed=None, use_gpu=False):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.h = h
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta = beta
        self.seed = seed
        self.use_gpu = use_gpu

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

            if self.verbose:
                print('Learning...')

            from .vaecf import learn

            res = learn(self.train_set, k=self.k, h=self.h, n_epochs=self.n_epochs,
                        batch_size=self.batch_size, learn_rate=self.learning_rate, beta=self.beta,
                        use_gpu=self.use_gpu, verbose=self.verbose, seed=self.seed)

            self.vae = res

            if self.verbose:
                print('Learning completed')
        elif self.verbose:
            print('%s is trained already (trainable = False)' % (self.name))

        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        import torch
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException("Can't make score prediction for (user_id=%d)" % user_idx)
            x_u = self.train_set.matrix[user_idx].copy()
            x_u.data = np.ones(len(x_u.data))
            z_u, _ = self.vae.encode(torch.tensor(x_u.A, dtype=torch.double))
            known_item_scores = self.vae.decode(z_u).data.cpu().numpy().flatten()
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(item_idx):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_idx, item_idx))
            x_u = self.train_set.matrix[user_idx].copy()
            x_u.data = np.ones(len(x_u.data))
            z_u, _ = self.vae.encode(torch.tensor(x_u.A, dtype=torch.double))
            user_pred = self.vae.decode(z_u).data.cpu().numpy().flatten()[item_idx]  # Fix me I am not efficient

            return user_pred
