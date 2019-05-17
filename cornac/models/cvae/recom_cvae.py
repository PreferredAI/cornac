# -*- coding: utf-8 -*-
"""
@author: Tran Thanh Binh
         
"""

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

    lambda_u: float, optional, default: 0.1
        The regularization hyper-parameter for user latent factor.

    lambda_v: float, optional, default: 10.0
        The regularization hyper-parameter for item latent factor.

    lambda_r: float, optional, default: 1.0
        Parameter that balance the focus on content or ratings

    lambda_w: float, optional, default: 2e-4
        The regularization for VAE weight

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
                 lambda_u=0.1, lambda_v=10, lambda_w=2e-4, lambda_r=1,
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

    @staticmethod
    def _build_data(csr_mat):
        data = []
        index_list = []
        rating_list = []
        for i in range(csr_mat.shape[0]):
            j, k = csr_mat.indptr[i], csr_mat.indptr[i + 1]
            index_list.append(csr_mat.indices[j:k])
            rating_list.append(csr_mat.data[j:k])
        data.append(index_list)
        data.append(rating_list)
        return data

    def _fit_cvae(self):

        user_data = self._build_data(self.train_set.matrix)
        item_data = self._build_data(self.train_set.matrix.T.tocsr())
        document = self.train_set.item_text.batch_bow(np.arange(self.train_set.num_items))  # bag of word feature
        document = (document - document.min()) / (document.max() - document.min())  # normalization

        # VAE initialization
        from .vae import VAE
        import tensorflow as tf
        from tqdm import trange

        self.vae = VAE(input_dim=self.input_dim, lambda_v=self.lambda_v, lambda_r=self.lambda_r, lambda_w=self.lambda_w,
                       n_z=self.n_z, layers=self.dimensions, loss_type=self.loss_type,
                       activations=self.act_fn, seed=self.seed, lr=self.lr)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())  # init variable

        loop = trange(self.n_epochs, disable=not self.verbose)
        for _ in loop:
            theta, gen_loss = self._vae_update(X_train=document, V=self.V)
            likelihood = self._cf_update(user_data=user_data, item_data=item_data, theta=theta)
            loss = -likelihood + 0.5 * gen_loss * self.train_set.num_items * self.lambda_r

            loop.set_postfix(loss=loss, likelihood=likelihood, gen_loss=gen_loss)

        tf.reset_default_graph()

    def _vae_update(self, X_train, V):
        for batch_ids in self.train_set.item_iter(batch_size=self.batch_size, shuffle=True):
            feed_dict = {self.vae.x: X_train[batch_ids],
                         self.vae.v: V[batch_ids]}
            self.sess.run(self.vae.optimizer, feed_dict=feed_dict)

        theta, gen_loss = self.sess.run([self.vae.z_mean, self.vae.gen_loss],
                                        feed_dict={self.vae.x: X_train})
        return theta, gen_loss

    def _cf_update(self, user_data, item_data, theta):
        n_user = len(user_data[0])
        n_item = len(item_data[0])

        # R_user and R_item contain rating values
        R_user = user_data[1]
        R_item = item_data[1]

        likelihood = 0
        VV = self.b * (self.V.T.dot(self.V)) + self.lambda_u * np.eye(self.n_z)

        # update user factors
        for i in range(n_user):
            idx_item = user_data[0][i]
            V_i = self.V[idx_item]
            R_i = R_user[i]
            A = VV + (self.a - self.b) * (V_i.T.dot(V_i))
            x = (self.a * V_i * (np.tile(R_i, (self.n_z, 1)).T)).sum(0)
            self.U[i] = np.linalg.solve(A, x)

            likelihood += -0.5 * self.lambda_u * np.dot(self.U[i], self.U[i])

        UU = self.b * (self.U.T.dot(self.U))

        # update item factors
        for j in range(n_item):
            idx_user = item_data[0][j]
            U_j = self.U[idx_user]
            R_j = R_item[j]

            UU_j = UU + (self.a - self.b) * (U_j.T.dot(U_j))
            A = UU_j + self.lambda_v * np.eye(self.n_z)
            x = (self.a * U_j * (np.tile(R_j, (self.n_z, 1)).T)).sum(0) + self.lambda_v * theta[j]
            self.V[j] = np.linalg.solve(A, x)

            likelihood += -0.5 * self.a * np.square(R_j).sum()
            likelihood += self.a * np.sum((U_j.dot(self.V[j])) * R_j)
            likelihood += - 0.5 * np.dot(self.V[j].dot(UU_j), self.V[j])

            ep = self.V[j, :] - theta[j, :]
            likelihood += -0.5 * self.lambda_v * np.sum(ep * ep)

        return likelihood

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
