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
from ...utils.init_utils import xavier_uniform


class CTR(Recommender):
    """Collaborative Topic Regression.

    Parameters
    ----------
    name: string, default: 'CTR'
        The name of the recommender model.

    k: int, optional, default: 200
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    lambda_u: float, optional, default: 0.01
        The regularization parameter for users.

    lambda_v: float, optional, default: 0.01
        The regularization parameter for items.

    a: float, optional, default: 1
        The confidence of observed ratings.

    b: float, optional, default: 0.01
        The confidence of unseen ratings.

    eta: float, optional, default: 0.01
        Added value for smoothing phi.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already 
        pre-trained (U and V are not None).

    init_params: dictionary, optional, default: None
        List of initial parameters, e.g., init_params = {'U':U, 'V':V}

        U: ndarray, shape (n_users,k)
            The user latent factors, optional initialization via init_params.
        V: ndarray, shape (n_items,k)
            The item latent factors, optional initialization via init_params.

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    Wang, Chong, and David M. Blei. "Collaborative topic modeling for recommending scientific articles."
    Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011.

    """

    def __init__(
        self,
        name="CTR",
        k=200,
        lambda_u=0.01,
        lambda_v=0.01,
        eta=0.01,
        a=1,
        b=0.01,
        max_iter=100,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.a = a
        self.b = b
        self.eta = eta
        self.name = name
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)
        self.V = self.init_params.get("V", None)

    def _init(self):
        rng = get_rng(self.seed)
        self.n_item = self.train_set.num_items
        self.n_user = self.train_set.num_users

        if self.U is None:
            self.U = xavier_uniform((self.n_user, self.k), rng)
        if self.V is None:
            self.V = xavier_uniform((self.n_item, self.k), rng)

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
            self._fit_ctr()

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

    def _fit_ctr(self,):
        from .ctr import Model

        user_data = self._build_data(self.train_set.matrix)
        item_data = self._build_data(self.train_set.matrix.T.tocsr())

        bow_mat = self.train_set.item_text.batch_bow(
            np.arange(self.n_item), keep_sparse=True
        )
        doc_ids, doc_cnt = self._build_data(bow_mat)  # bag of word feature

        self.model = Model(
            n_user=self.n_user,
            n_item=self.n_item,
            U=self.U,
            V=self.V,
            k=self.k,
            n_vocab=self.train_set.item_text.vocab.size,
            lambda_u=self.lambda_u,
            lambda_v=self.lambda_v,
            a=self.a,
            b=self.b,
            max_iter=self.max_iter,
            seed=self.seed,
        )

        loop = trange(self.max_iter, disable=not self.verbose)
        for _ in loop:
            cf_loss = self.model.update_cf(
                user_data=user_data, item_data=item_data
            )  # u and v updating
            lda_loss = self.model.update_theta(doc_ids=doc_ids, doc_cnt=doc_cnt)
            self.model.update_beta()
            loop.set_postfix(cf_loss=cf_loss, lda_likelihood=-lda_loss)

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

            known_item_scores = self.V.dot(self.U[user_idx, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            user_pred = self.V[item_idx, :].dot(self.U[user_idx, :])
            return user_pred
