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
from ..recommender import ANNMixin, MEASURE_DOT
from ...utils.common import scale
from ...exception import ScoreException


class BiVAECF(Recommender, ANNMixin):
    """Bilateral Variational AutoEncoder for Collaborative Filtering.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the stochastic user ``theta'' and item ``beta'' factors.

    encoder_structure: list, default: [20]
        The number of neurons per layer of the user and item encoders for BiVAE.
        For example, encoder_structure = [20], the user (item) encoder structure will be [num_items, 20, k] ([num_users, 20, k]).

    act_fn: str, default: 'tanh'
        Name of the activation function used between hidden layers of the auto-encoder.
        Supported functions: ['sigmoid', 'tanh', 'elu', 'relu', 'relu6']

    likelihood: str, default: 'pois'
        The likelihood function used for modeling the observations.
        Supported choices:

        bern: Bernoulli likelihood
        gaus: Gaussian likelihood
        pois: Poisson likelihood

    n_epochs: int, optional, default: 100
        The number of epochs for SGD.

    batch_size: int, optional, default: 100
        The batch size.

    learning_rate: float, optional, default: 0.001
        The learning rate for Adam.

    beta_kl: float, optional, default: 1.0
        The weight of the KL terms as in beta-VAE.

    cap_priors: dict, optional, default: {"user":False, "item":False}
        When {"user":True, "item":True}, CAP priors are used (see BiVAE paper for details),\
        otherwise the standard Normal is used as a Prior over the user and item latent variables.

    name: string, optional, default: 'BiVAECF'
        The name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model is already \
        pre-trained.

    verbose: boolean, optional, default: False
        When True, some running logs are displayed.

    seed: int, optional, default: None
        Random seed for parameters initialization.

    use_gpu: boolean, optional, default: True
        If True and your system supports CUDA then training is performed on GPUs.

    References
    ----------
    * Quoc-Tuan Truong, Aghiles Salah, Hady W. Lauw. " Bilateral Variational Autoencoder for Collaborative Filtering."
    ACM International Conference on Web Search and Data Mining (WSDM). 2021.
    """

    def __init__(
        self,
        name="BiVAECF",
        k=10,
        encoder_structure=[20],
        act_fn="tanh",
        likelihood="pois",
        n_epochs=100,
        batch_size=100,
        learning_rate=0.001,
        beta_kl=1.0,
        cap_priors={"user": False, "item": False},
        trainable=True,
        verbose=False,
        seed=None,
        use_gpu=True,
    ):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.encoder_structure = encoder_structure
        self.act_fn = act_fn
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta_kl = beta_kl
        self.cap_priors = cap_priors
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
        from .bivae import BiVAE, learn

        self.device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        if self.trainable:
            feature_dim = {"user": None, "item": None}
            if self.cap_priors.get("user", False):
                if train_set.user_feature is None:
                    raise ValueError(
                        "CAP priors for users is set to True but no user features are provided"
                    )
                else:
                    feature_dim["user"] = train_set.user_feature.feature_dim

            if self.cap_priors.get("item", False):
                if train_set.item_feature is None:
                    raise ValueError(
                        "CAP priors for items is set to True but no item features are provided"
                    )
                else:
                    feature_dim["item"] = train_set.item_feature.feature_dim

            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

            if not hasattr(self, "bivae"):
                num_items = train_set.matrix.shape[1]
                num_users = train_set.matrix.shape[0]
                self.bivae = BiVAE(
                    k=self.k,
                    user_encoder_structure=[num_items] + self.encoder_structure,
                    item_encoder_structure=[num_users] + self.encoder_structure,
                    act_fn=self.act_fn,
                    likelihood=self.likelihood,
                    cap_priors=self.cap_priors,
                    feature_dim=feature_dim,
                    batch_size=self.batch_size,
                ).to(self.device)

            learn(
                self.bivae,
                train_set,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learn_rate=self.learning_rate,
                beta_kl=self.beta_kl,
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
        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)

        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        if item_idx is None:
            theta_u = self.bivae.mu_theta[user_idx].view(1, -1)
            beta = self.bivae.mu_beta
            return self.bivae.decode_user(theta_u, beta).cpu().numpy().ravel()
        else:
            theta_u = self.bivae.mu_theta[user_idx].view(1, -1)
            beta_i = self.bivae.mu_beta[item_idx].view(1, -1)
            pred = self.bivae.decode_user(theta_u, beta_i).cpu().numpy().ravel()
            return scale(pred, self.min_rating, self.max_rating, 0.0, 1.0)

    def get_vector_measure(self):
        """Getting a valid choice of vector measurement in ANNMixin._measures.

        Returns
        -------
        measure: MEASURE_DOT
            Dot product aka. inner product
        """
        return MEASURE_DOT

    def get_user_vectors(self):
        """Getting a matrix of user vectors serving as query for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of user vectors for all users available in the model.
        """
        user_vectors = self.bivae.mu_theta.detach().cpu().numpy()
        return user_vectors

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        out: numpy.array
            Matrix of item vectors for all items available in the model.
        """
        item_vectors = self.bivae.mu_beta.detach().cpu().numpy()
        return item_vectors

    def save(self, save_dir=None, save_trainset=True):
        """Save model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

        save_trainset: bool, default: True
            Save train_set together with the model. This is useful
            if we want to deploy model later because train_set is
            required for certain evaluation steps.

        Returns
        -------
        model_file : str
            Path to the model file stored on the filesystem.
        """
        import torch

        if save_dir is None:
            return

        self.bivae.to(torch.device("cpu"))
        model_file = Recommender.save(
            self, save_dir=save_dir, save_trainset=save_trainset
        )

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load model from the filesystem.

        Parameters
        ----------
        model_path: str, required
            Path to a file or directory where the model is stored. If a directory is
            provided, the latest model will be loaded.

        trainable: boolean, optional, default: False
            Set it to True if you would like to finetune the model. By default,
            the model parameters are assumed to be fixed after being loaded.

        Returns
        -------
        self : object
        """
        import torch

        model = Recommender.load(model_path, trainable)
        if "cuda" in str(model.device) and torch.cuda.is_available():
            model.bivae.to(model.device)

        return model
