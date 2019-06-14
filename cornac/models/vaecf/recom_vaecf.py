# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np
from ..recommender import Recommender
from ...utils.common import sigmoid
from ...utils.common import scale
from ...exception import ScoreException
import torch


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

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD_RMSProp.

    gamma: float, optional, default: 0.9
        The weight for previous/current gradient in RMSProp.

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

    def __init__(self, name="PVAE", k=10, h_dim=20, n_epochs=100, batch_size=100, learning_rate=0.001, gamma=0.9,
                 trainable=True, verbose=False, use_gpu=False, init_params={}, seed=None):
        Recommender.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.h_dim = h_dim
        self.batch_size = batch_size
        self.init_params = init_params
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.use_gpu = use_gpu
        self.loss = np.full(n_epochs, 0)
        self.eps = 1e-10
        self.seed = seed

    # fit the recommender model to the traning data
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

        if self.trainable:

            if self.verbose:
                print('Learning...')

            from .pvae import learn
            
            self.features = train_set.user_graph.matrix[:self.train_set.num_users, :self.train_set.num_users]
            
            #self.features = torch.tensor(self.features.A,dtype=torch.double, requires_grad=False)
            #self.features = torch.randn_like(self.features)*1.
            
            res = learn(self.train_set, self.features, k=self.k, h_dim=self.h_dim, n_epochs=self.n_epochs,
                      batch_size=self.batch_size, learn_rate=self.learning_rate, gamma=self.gamma,
                      init_params=self.init_params, use_gpu=self.use_gpu, verbose=self.verbose, seed=self.seed)

            self.vae, self.disc, self.kl_vals, self.f_vals = res

            z_u, _ = self.vae.encode(torch.tensor(self.train_set.matrix.A, dtype=torch.double))
            self.theta = z_u.data

            if self.verbose:
                print('Learning completed')
        elif self.verbose:
            print('%s is trained already (trainable = False)' % (self.name))

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
            r_u = self.train_set.matrix[user_id].copy()
            r_u.data = np.ones(len(r_u.data))
            z_u, _ = self.vae.encode(torch.tensor(r_u.A,dtype=torch.double))
            known_item_scores = self.vae.decode(z_u).data.cpu().numpy().flatten()
            # known_item_scores = np.asarray(np.zeros(self.V.shape[0]),dtype='float32')
            # fast_dot(self.U[user_id], self.V, known_item_scores)
            # fast_dot(self.P[user_id], self.Q, known_item_scores)
            return known_item_scores
        else:
            if self.train_set.is_unk_user(user_id) or self.train_set.is_unk_item(item_id):
                raise ScoreException("Can't make score prediction for (user_id=%d, item_id=%d)" % (user_id, item_id))
            user_pred = self.V[item_id, :].dot(self.U[user_id, :]) + self.Q[item_id, :].dot(self.P[user_id, :])
            user_pred = sigmoid(user_pred)

            user_pred = scale(user_pred, self.train_set.min_rating, self.train_set.max_rating, 0., 1.)

            return user_pred
