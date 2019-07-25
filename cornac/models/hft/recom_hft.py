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


class HFT(Recommender):
    """Collaborative Topic Regression

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

    def __init__(self, name='HFT', k=10, lambda_reg=0.01, latent_reg=0.1, vocab_size=8000,
                 max_iter=50, grad_iter=50, trainable=True, verbose=True, init_params=None,
                 seed=None):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.lambda_reg = lambda_reg
        self.latent_reg = latent_reg
        self.grad_iter = grad_iter
        self.name = name
        self.max_iter = max_iter
        self.verbose = verbose
        self.init_params = {} if not init_params else init_params
        self.seed = seed
        self.vocab_size = vocab_size

    def fit(self, train_set):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: object of type TrainSet, required
            An object contraining the user-item preference in csr scipy sparse format,\
            as well as some useful attributes such as mappings to the original user/item ids.\
            Please refer to the class TrainSet in the "data" module for details.
        """
        Recommender.fit(self, train_set)
        from ...utils.init_utils import uniform

        self.n_item = self.train_set.num_items
        self.n_user = self.train_set.num_users

        n_params = 1 + 1 + self.n_user + self.n_item + self.k * (self.n_user + self.n_item + self.vocab_size)
        self.params = self.init_params.get('params', uniform((n_params, 1), random_state=self.seed))
        self._update_view()
        self.alpha = self.train_set.global_mean

        if self.trainable:
            self._fit_hft()

    def _update_view(self):

        params_length = np.array(
            [1, 1, self.n_user, self.n_item, self.n_user * self.k, self.n_item * self.k, self.vocab_size * self.k])
        idx = params_length.cumsum()
        # create parameter view
        self.alpha = self.params[0:idx[0], ]
        self.kappa = self.params[idx[0]:idx[1], ]
        self.bias_user = self.params[idx[1]:idx[2], ]
        self.bias_item = self.params[idx[2]:idx[3], ]
        self.U = self.params[idx[3]:idx[4], ].reshape(self.n_user, self.k)
        self.V = self.params[idx[4]:idx[5], ].reshape(self.n_item, self.k)
        self.topic_words = self.params[idx[5]:idx[6], ].reshape(self.vocab_size, self.k)

    def _fit_hft(self):

        from .hft import Model
        from tqdm import trange

        model = Model(n_user=self.n_user, n_item=self.n_item, params=self.params,
                      n_vocab=self.vocab_size, k=self.k, lambda_reg=self.lambda_reg,
                      latent_reg=self.latent_reg, grad_iter=self.grad_iter)

        bow_mat = self.train_set.item_text.batch_bow(np.arange(self.n_item), keep_sparse=True)
        documents, _ = self._build_data(bow_mat)  # bag of word feature
        # Rating data
        user_data = self._build_data(self.train_set.matrix)
        item_data = self._build_data(self.train_set.matrix.T.tocsr())
        # randomly assign word topic
        model.random_int_topic(docs=documents)
        # training
        loop = trange(self.max_iter, disable=not self.verbose)
        for _ in loop:
            model.assign_word_topics(docs=documents)
            loss = model.update_params(rating_data=(user_data, item_data))
            loop.set_postfix(loss=loss)

        self.params = model.get_parameter()
        self._update_view()

        if self.verbose:
            print('Learning completed!')

    @staticmethod
    def _build_data(csr_mat):
        index_list = []
        rating_list = []
        for i in range(csr_mat.shape[0]):
            j, k = csr_mat.indptr[i], csr_mat.indptr[i + 1]
            index_list.append(csr_mat.indices[j:k])
            rating_list.append(csr_mat.data[j:k])
        return index_list, rating_list

    def score(self, user_id, item_id=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform score prediction.

        item_id: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if item_id is None:
            if self.train_set.is_unk_user(user_id):
                raise ScoreException("Can't make score prediction for (user_id=%d)" % user_id)

            known_item_scores = self.alpha.item() + self.bias_user[user_id] + self.bias_item + self.V.dot(
                self.U[user_id, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))

            user_pred = self.alpha.item() + self.bias_user[user_id] + self.bias_item[item_id] + self.V[item_id, :].dot(
                self.U[user_id, :])

            return user_pred
