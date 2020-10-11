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

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng
from ...utils.init_utils import normal


class HFT(Recommender):
    """Hidden Factors and Hidden Topics

    Parameters
    ----------
    name: string, default: 'HFT'
        The name of the recommender model.

    k: int, optional, default: 10
        The dimension of the latent factors.

    max_iter: int, optional, default: 50
        Maximum number of iterations for EM.

    grad_iter: int, optional, default: 50
        Maximum number of iterations for L-BFGS.

    lambda_text: float, default: 0.1
        Weight of corpus likelihood in objective function.

    l2_reg: float, default: 0.001
        Regularization for user item latent factors.

    vocab_size: int, optional, default: 8000
        Size of vocabulary for review text.

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'alpha': alpha, 'beta_u': beta_u,
        'beta_i': beta_i, 'gamma_u': gamma_u, 'gamma_v': gamma_v}

        alpha: float
            Model offset, optional initialization via init_params.

        beta_u: ndarray. shape (n_user, 1)
            User biases, optional initialization via init_params.

        beta_u: ndarray. shape (n_item, 1)
            Item biases, optional initialization via init_params.

        gamma_u: ndarray, shape (n_users,k)
            The user latent factors, optional initialization via init_params.

        gamma_v: ndarray, shape (n_items,k)
            The item latent factors, optional initialization via init_params.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, some running logs are displayed.
        
    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    Julian McAuley, Jure Leskovec. "Hidden Factors and Hidden Topics: Understanding Rating Dimensions with Review Text"
    RecSys '13 Proceedings of the 7th ACM conference on Recommender systems Pages 165-172
    """

    def __init__(
        self,
        name="HFT",
        k=10,
        max_iter=50,
        grad_iter=50,
        lambda_text=0.1,
        l2_reg=0.001,
        vocab_size=8000,
        init_params=None,
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.k = k
        self.lambda_text = lambda_text
        self.l2_reg = l2_reg
        self.grad_iter = grad_iter
        self.name = name
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        self.vocab_size = vocab_size

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.alpha = self.init_params.get("alpha", None)
        self.beta_u = self.init_params.get("beta_u", None)
        self.beta_i = self.init_params.get("beta_i", None)
        self.gamma_u = self.init_params.get("gamma_u", None)
        self.gamma_i = self.init_params.get("gamma_i", None)

    def _init(self):
        rng = get_rng(self.seed)
        self.n_item = self.train_set.num_items
        self.n_user = self.train_set.num_users

        if self.alpha is None:
            self.alpha = self.train_set.global_mean
        if self.beta_u is None:
            self.beta_u = normal(self.n_user, std=0.01, random_state=rng)
        if self.beta_i is None:
            self.beta_i = normal(self.n_item, std=0.01, random_state=rng)
        if self.gamma_u is None:
            self.gamma_u = normal((self.n_user, self.k), std=0.01, random_state=rng)
        if self.gamma_i is None:
            self.gamma_i = normal((self.n_item, self.k), std=0.01, random_state=rng)

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

        self._init()

        if self.trainable:
            self._fit_hft()

        return self

    @staticmethod
    def _build_data(csr_mat):
        index_list = []
        rating_list = []
        for i in range(csr_mat.shape[0]):
            j, k = csr_mat.indptr[i], csr_mat.indptr[i + 1]
            index_list.append(csr_mat.indices[j:k])
            rating_list.append(csr_mat.data[j:k])
        return index_list, rating_list

    def _fit_hft(self):
        from .hft import Model

        # document data
        bow_mat = self.train_set.item_text.batch_bow(
            np.arange(self.n_item), keep_sparse=True
        )
        documents, _ = self._build_data(bow_mat)  # bag of word feature
        # Rating data
        user_data = self._build_data(self.train_set.matrix)
        item_data = self._build_data(self.train_set.matrix.T.tocsr())

        model = Model(
            n_user=self.n_user,
            n_item=self.n_item,
            alpha=self.alpha,
            beta_u=self.beta_u,
            beta_i=self.beta_i,
            gamma_u=self.gamma_u,
            gamma_i=self.gamma_i,
            n_vocab=self.vocab_size,
            k=self.k,
            lambda_text=self.lambda_text,
            l2_reg=self.l2_reg,
            grad_iter=self.grad_iter,
        )

        model.init_count(docs=documents)

        # training
        loop = trange(self.max_iter, disable=not self.verbose)
        for _ in loop:
            model.assign_word_topics(docs=documents)
            loss = model.update_params(rating_data=(user_data, item_data))
            loop.set_postfix(loss=loss)

        self.alpha, self.beta_u, self.beta_i, self.gamma_u, self.gamma_i = (
            model.get_parameter()
        )

        if self.verbose:
            print("Learning completed!")

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

            known_item_scores = (
                self.alpha
                + self.beta_u[user_idx]
                + self.beta_i
                + self.gamma_i.dot(self.gamma_u[user_idx, :])
            )
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = (
                self.alpha
                + self.beta_u[user_idx]
                + self.beta_i[item_idx]
                + self.gamma_i[item_idx, :].dot(self.gamma_u[user_idx, :])
            )

            return user_pred
