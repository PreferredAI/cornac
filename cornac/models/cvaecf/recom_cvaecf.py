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


class CVAECF(Recommender):
    """Conditional Variational Autoencoder for Collaborative Filtering.

    Parameters
    ----------
    z_dim: int, optional, default: 20
        The dimension of the stochastic user factors ``z'' representing the preference information.

    h_dim: int, optional, default: 20
        The dimension of the stochastic user factors ``h'' representing the auxiliary data.

    autoencoder_structure: list, default: [20]
        The number of neurons of encoder/decoder hidden layer for CVAE.
        For example, when autoencoder_structure = [20],
        the CVAE encoder structures will be [y_dim, 20, z_dim] and [x_dim, 20, h_dim],
        the decoder structure will be [z_dim + h_dim, 20, y_dim], where y and x are respectively the preference and \
        auxiliary data.

    act_fn: str, default: 'tanh'
        Name of the activation function used between hidden layers of the auto-encoder.
        Supported functions: ['sigmoid', 'tanh', 'elu', 'relu', 'relu6']

    likelihood: str, default: 'mult'
        Name of the likelihood function used for modeling user preferences.
        Supported choices:

        mult: Multinomial likelihood
        bern: Bernoulli likelihood
        gaus: Gaussian likelihood
        pois: Poisson likelihood

    n_epochs: int, optional, default: 100
        The number of epochs for SGD.

    batch_size: int, optional, default: 128
        The batch size.

    learning_rate: float, optional, default: 0.001
        The learning rate for Adam.

    beta: float, optional, default: 1.0
        The weight of the KL term KL(q(z|y)||p(z)) as in beta-VAE.

    alpha_1: float, optional, default: 1.0
        The weight of the KL term KL(q(h|x)||p(h|x)).

    alpha_2: float, optional, default: 1.0
        The weight of the KL term KL(q(h|x)||q(h|y)).

    name: string, optional, default: 'CVAECF'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained, and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    use_gpu: boolean, optional, default: False
        If True and your system supports CUDA then training is performed on GPUs.

    user auxiliary data : See "cornac/examples/cvaecf_filmtrust.py" for an example of how to use \
        cornac's graph modality to load and provide the ``user network'' for CVAECF.

    References
    ----------
    * Lee, Wonsung, Kyungwoo Song, and Il-Chul Moon. "Augmented variational autoencoders for collaborative filtering \
     with auxiliary information." Proceedings of ACM CIKM. 2017.
    """

    def __init__(
            self,
            name="CVAECF",
            z_dim=20,
            h_dim=20,
            autoencoder_structure=[20],
            act_fn="tanh",
            likelihood="mult",
            n_epochs=100,
            batch_size=128,
            learning_rate=0.001,
            beta=1.0,
            alpha_1=1.0,
            alpha_2=1.0,
            trainable=True,
            verbose=False,
            seed=None,
            use_gpu=False,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.autoencoder_structure = autoencoder_structure
        self.act_fn = act_fn
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta = beta
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
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

        import torch
        from .cvaecf import CVAE, learn

        self.device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        if self.trainable:
            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

            if not hasattr(self, "cvae"):
                n_items = train_set.matrix.shape[1]
                n_users = train_set.matrix.shape[0]
                self.cvae = CVAE(
                    self.z_dim,
                    self.h_dim,
                    [n_items] + self.autoencoder_structure,
                    [n_users] + self.autoencoder_structure,
                    self.act_fn,
                    self.likelihood,
                ).to(self.device)

            learn(
                self.cvae,
                self.train_set,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learn_rate=self.learning_rate,
                beta=self.beta,
                alpha_1=self.alpha_1,
                alpha_2=self.alpha_2,
                verbose=self.verbose,
                device=self.device,
            )

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

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
        import torch

        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            y_u = self.train_set.matrix[user_idx].copy()
            y_u.data = np.ones(len(y_u.data))
            y_u = torch.tensor(y_u.A, dtype=torch.float32, device=self.device)
            z_u, _ = self.cvae.encode_qz(y_u)

            x_u = self.train_set.user_graph.matrix[user_idx].copy()
            x_u.data = np.ones(len(x_u.data))
            x_u = torch.tensor(x_u.A, dtype=torch.float32, device=self.device)
            h_u, _ = self.cvae.encode_qhx(x_u)

            known_item_scores = self.cvae.decode(z_u, h_u).data.cpu().numpy().flatten()

            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                    item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            y_u = self.train_set.matrix[user_idx].copy()
            y_u.data = np.ones(len(y_u.data))
            y_u = torch.tensor(y_u.A, dtype=torch.float32, device=self.device)
            z_u, _ = self.cvae.encode_qz(y_u)

            x_u = self.train_set.user_graph.matrix[user_idx].copy()
            x_u.data = np.ones(len(x_u.data))
            x_u = torch.tensor(x_u.A, dtype=torch.float32, device=self.device)
            h_u, _ = self.cvae.encode_qhx(x_u)

            user_pred = self.cvae.decode(z_u, h_u).data.cpu().numpy().flatten()[item_idx]

            return user_pred
