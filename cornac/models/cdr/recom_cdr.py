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


class CDR(Recommender):
    """Collaborative Deep Ranking.

    Parameters
    ----------
    k: int, optional, default: 50
        The dimension of the latent factors.

    max_iter: int, optional, default: 100
        Maximum number of iterations or the number of epochs for SGD.

    autoencoder_structure: list, default: None
        The number of neurons of encoder/decoder layer for SDAE.
        For example, autoencoder_structure = [200], the SDAE structure will be [vocab_size, 200, k, 200, vocab_size]

    act_fn: str, default: 'relu'
        Name of the activation function used for the auto-encoder.
        Supported functions: ['sigmoid', 'tanh', 'elu', 'relu', 'relu6', 'leaky_relu', 'identity']

    learning_rate: float, optional, default: 0.001
        The learning rate for AdamOptimizer.

    lambda_u: float, optional, default: 0.1
        The regularization parameter for users.

    lambda_v: float, optional, default: 10
        The regularization parameter for items.

    lambda_w: float, optional, default: 0.1
        The regularization parameter for SDAE weights.

    lambda_n: float, optional, default: 1000
        The regularization parameter for SDAE output.

    corruption_rate: float, optional, default: 0.3
        The corruption ratio for SDAE.

    dropout_rate: float, optional, default: 0.1
        The probability that each element is removed in dropout of SDAE.

    batch_size: int, optional, default: 128
        The batch size for SGD.

    name: string, optional, default: 'CDR'
        The name of the recommender model.

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
    Collaborative Deep Ranking: A Hybrid Pair-Wise Recommendation Algorithm with Implicit Feedback
    Ying H., Chen L., Xiong Y., Wu J. (2016)

    """

    def __init__(
        self,
        name="CDR",
        k=50,
        autoencoder_structure=None,
        act_fn="relu",
        lambda_u=0.1,
        lambda_v=100,
        lambda_w=0.1,
        lambda_n=1000,
        corruption_rate=0.3,
        learning_rate=0.001,
        dropout_rate=0.1,
        batch_size=128,
        max_iter=100,
        trainable=True,
        verbose=True,
        vocab_size=8000,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w
        self.lambda_n = lambda_n
        self.corruption_rate = corruption_rate
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.name = name
        self.max_iter = max_iter
        self.autoencoder_structure = autoencoder_structure
        self.act_fn = act_fn
        self.batch_size = batch_size
        self.verbose = verbose
        self.vocab_size = vocab_size
        self.seed = seed
        self.rng = get_rng(seed)

        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.U = self.init_params.get("U", None)
        self.V = self.init_params.get("V", None)

    def _init(self):
        n_users, n_items = self.train_set.num_users, self.train_set.num_items

        if self.U is None:
            self.U = xavier_uniform((n_users, self.k), self.rng)
        if self.V is None:
            self.V = xavier_uniform((n_items, self.k), self.rng)

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
            self._fit_cdr()

        return self

    def _fit_cdr(self):
        import tensorflow as tf
        from .model import Model

        n_users = self.train_set.num_users
        n_items = self.train_set.num_items

        text_feature = self.train_set.item_text.batch_bow(
            np.arange(n_items)
        )  # bag of word feature
        text_feature = (text_feature - text_feature.min()) / (
            text_feature.max() - text_feature.min()
        )  # normalization

        # Build model
        layer_sizes = (
            [self.vocab_size]
            + self.autoencoder_structure
            + [self.k]
            + self.autoencoder_structure
            + [self.vocab_size]
        )
        tf.set_random_seed(self.seed)
        model = Model(
            n_users=n_users,
            n_items=n_items,
            n_vocab=self.vocab_size,
            k=self.k,
            layers=layer_sizes,
            lambda_u=self.lambda_u,
            lambda_v=self.lambda_v,
            lambda_w=self.lambda_w,
            lambda_n=self.lambda_n,
            lr=self.learning_rate,
            dropout_rate=self.dropout_rate,
            U=self.U,
            V=self.V,
            act_fn=self.act_fn,
            seed=self.seed,
        )

        # Training model
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            loop = trange(self.max_iter, disable=not self.verbose)
            for _ in loop:
                corruption_mask = self.rng.binomial(
                    1, 1 - self.corruption_rate, (n_items, self.vocab_size)
                )
                sum_loss = 0
                count = 0
                batch_count = 0
                for batch_u, batch_i, batch_j in self.train_set.uij_iter(
                    batch_size=self.batch_size, shuffle=True
                ):
                    feed_dict = {
                        model.mask_input: corruption_mask[batch_i, :],
                        model.text_input: text_feature[batch_i, :],
                        model.batch_u: batch_u,
                        model.batch_i: batch_i,
                        model.batch_j: batch_j,
                    }

                    sess.run(model.opt1, feed_dict)  # train U, V
                    _, _loss = sess.run(
                        [model.opt2, model.loss], feed_dict
                    )  # train SDAE

                    sum_loss += _loss
                    count += len(batch_u)
                    batch_count += 1
                    if batch_count % 10 == 0:
                        loop.set_postfix(loss=(sum_loss / count))

            self.U, self.V = sess.run([model.U, model.V])

        tf.reset_default_graph()

        if self.verbose:
            print("\nLearning completed")

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
