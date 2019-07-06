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


class CVAE(Recommender):
    """
    Collaborative Variational Autoencoder

    Parameters
    ----------
    z_dim: int, optional, default: 50
        The dimension of the user and item latent factors.

    n_epochs: int, optional, default: 100
        Maximum number of epochs for training.

    lambda_u: float, optional, default: 1e-4
        The regularization hyper-parameter for user latent factor.

    lambda_v: float, optional, default: 0.001
        The regularization hyper-parameter for item latent factor.

    lambda_r: float, optional, default: 10.0
        Parameter that balance the focus on content or ratings

    lambda_w: float, optional, default: 1e-4
        The regularization for VAE weights

    lr: float, optional, default: 0.001
        Learning rate in the auto-encoder training

    a: float, optional, default: 1
        The confidence of observed ratings.

    b: float, optional, default: 0.01
        The confidence of unseen ratings.

    input_dim: int, optional, default: 8000
        The size of input vector

    vae_layers: list, optional, default: [200, 100]
        The list containing size of each layers in neural network structure

    act_fn: str, default: 'sigmoid'
        Name of the activation function used for the variational auto-encoder.
        Supported functions: ['sigmoid', 'tanh', 'elu', 'relu', 'relu6', 'leaky_relu', 'identity']

    loss_type: String, optional, default: "cross-entropy"
        Either "cross-entropy" or "rmse"
        The type of loss function in the last layer

    batch_size: int, optional, default: 128
        The batch size for SGD.

    init_params: dict, optional, default: {'U':None, 'V':None}
        Initial U and V latent matrix

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).

    References
    ----------
    Collaborative Variational Autoencoder for Recommender Systems
    X. Li and J. She ACM SIGKDD International Conference on Knowledge Discovery and Data Mining 2017

    http://eelxpeng.github.io/assets/paper/Collaborative_Variational_Autoencoder.pdf

    """

    def __init__(self, name="CVAE", z_dim=50, n_epochs=100,
                 lambda_u=1e-4, lambda_v=0.001, lambda_r=10, lambda_w=1e-4,
                 lr=0.001, a=1, b=0.01, input_dim=8000,
                 vae_layers=[200, 100], act_fn='sigmoid', loss_type='cross-entropy',
                 batch_size=128, init_params=None, trainable=True, seed=None, verbose=True):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_r = lambda_r
        self.lambda_w = lambda_w
        self.a = a
        self.b = b
        self.n_epochs = n_epochs
        self.input_dim = input_dim
        self.dimensions = vae_layers
        self.n_z = z_dim
        self.loss_type = loss_type
        self.act_fn = act_fn
        self.lr = lr
        self.batch_size = batch_size
        self.init_params = {} if not init_params else init_params
        self.seed = seed

    def fit(self, train_set):
        """Fit the model.

        Parameters
        ----------
        train_set: :obj:`cornac.data.MultimodalTrainSet`
            Multimodal training set.

        """
        Recommender.fit(self, train_set)

        from ...utils import get_rng
        from ...utils.init_utils import xavier_uniform

        rng = get_rng(self.seed)
        self.U = self.init_params.get('U', xavier_uniform((self.train_set.num_users, self.n_z), rng))
        self.V = self.init_params.get('V', xavier_uniform((self.train_set.num_items, self.n_z), rng))

        if self.trainable:
            self._fit_cvae()

    def _fit_cvae(self):
        R = self.train_set.csc_matrix  # csc for efficient slicing over items
        document = self.train_set.item_text.batch_bow(np.arange(self.train_set.num_items))  # bag of word feature
        document = (document - document.min()) / (document.max() - document.min())  # normalization

        # VAE initialization
        from .cvae import Model
        import tensorflow as tf
        from tqdm import trange

        model = Model(n_users=self.train_set.num_users, n_items=self.train_set.num_items,
                      input_dim=self.input_dim, U=self.U, V=self.V, n_z=self.n_z,
                      lambda_u=self.lambda_u, lambda_v=self.lambda_v, lambda_r=self.lambda_r,
                      lambda_w=self.lambda_w, layers=self.dimensions, loss_type=self.loss_type,
                      act_fn=self.act_fn, seed=self.seed, lr=self.lr)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())  # init variable

        loop = trange(self.n_epochs, disable=not self.verbose)
        for _ in loop:
            cf_loss, vae_loss, count = 0, 0, 0
            for i, batch_ids in enumerate(self.train_set.item_iter(self.batch_size, shuffle=True)):
                batch_R = R[:, batch_ids]
                batch_C = np.ones(batch_R.shape) * self.b
                batch_C[batch_R.nonzero()] = self.a

                feed_dict = {model.x: document[batch_ids],
                             model.ratings: batch_R.A,
                             model.C: batch_C,
                             model.item_ids: batch_ids}
                _, _vae_los = sess.run([model.vae_update, model.vae_loss], feed_dict)
                _, _cf_loss = sess.run([model.cf_update, model.cf_loss], feed_dict)

                cf_loss += _cf_loss
                vae_loss += _vae_los
                count += len(batch_ids)
                if i % 10 == 0:
                    loop.set_postfix(vae_loss=(vae_loss / count),
                                     cf_loss=(cf_loss / count))

        self.U, self.V = sess.run([model.U, model.V])

        tf.reset_default_graph()

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

            known_item_scores = self.V.dot(self.U[user_id, :])
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))

            user_pred = self.V[item_id, :].dot(self.U[user_id, :])

            return user_pred
