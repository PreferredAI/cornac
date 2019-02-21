# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao <jyguo@smu.edu.sg>
         Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..recommender import Recommender
import numpy as np

from ...utils.common import intersects
from ...utils import tryimport
torch = tryimport('torch')
tqdm = tryimport('tqdm')


class VBPR(Recommender):
    """Visual Bayesian Personalized Ranking.

    Parameters
    ----------
    k: int, optional, default: 5
        The dimension of the gamma latent factors.

    d: int, optional, default: 5
        The dimension of the theta latent factors.

    n_epochs: int, optional, default: 20
        Maximum number of epochs for SGD.

    batch_size: int, optional, default: 100
        The batch size for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    lambda_t: float, optional, default: 0.01
        The lambda theta regularization hyper-parameter.

    lambda_b: float, optional, default: 0.01
        The lambda beta regularization hyper-parameter.

    lambda_e: float, optional, default: 0.0
        The regularization hyper-parameter for embedding matrix E.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).

    References
    ----------
    * HE, Ruining et MCAULEY, Julian. VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. In : AAAI. 2016. p. 144-150.
    """

    def __init__(self,
                 k=10, d=10,
                 n_epochs=20, batch_size=100, learning_rate=0.001,
                 lambda_t=0.01, lambda_b=0.01, lambda_e=0.0,
                 use_gpu=False, trainable=True, **kwargs):
        Recommender.__init__(self, name='VBPR', trainable=trainable)
        self.k = k
        self.d = d
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_t = lambda_t
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e

        # Initial params
        self.beta_item = kwargs.get('beta_item', None)
        self.gamma_user = kwargs.get('gamma_user', None)
        self.gamma_item = kwargs.get('gamma_item', None)
        self.theta_user = kwargs.get('theta_user', None)
        self.emb_matrix = kwargs.get('emb_matrix', None)
        self.beta_prime = kwargs.get('beta_prime', None)

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def _load_or_randn(self, size, init_values=None):
        if init_values is None:
            tensor = torch.randn(size, requires_grad=True, device=self.device)
        else:
            tensor = torch.tensor(init_values, requires_grad=True, device=self.device)
        return tensor

    def _init_params(self, n_users, n_items, feature_dim):
        Bi = self._load_or_randn((n_items), init_values=self.beta_item)
        Gu = self._load_or_randn((n_users, self.k), init_values=self.gamma_user)
        Gi = self._load_or_randn((n_items, self.k), init_values=self.gamma_item)
        Tu = self._load_or_randn((n_users, self.d), init_values=self.theta_user)
        E = self._load_or_randn((feature_dim, self.d), init_values=self.emb_matrix)
        Bp = self._load_or_randn((n_users, feature_dim), init_values=self.beta_prime)

        return Bi, Gu, Gi, Tu, E, Bp

    def _l2_loss(self, *tensors):
        l2_loss = 0
        for tensor in tensors:
            l2_loss += torch.sum(tensor ** 2) / 2
        return l2_loss

    def fit(self, train_set):
        """Fit the model.

        Parameters
        ----------
        train_set: :obj:`cornac.data.MultimodalTrainSet`
            Multimodal training set.

        """
        Recommender.fit(self, train_set)

        if not self.trainable:
            print('%s is trained already (trainable = False)' % (self.name))
            return

        # Item visual feature from CNN
        self.item_feature = train_set.item_image.data_feature[:self.train_set.num_items]
        F = torch.from_numpy(self.item_feature).float().to(self.device)

        # Learned parameters
        Bi, Gu, Gi, Tu, E, Bp = self._init_params(n_users=train_set.num_users,
                                                      n_items=train_set.num_items,
                                                      feature_dim=train_set.item_image.feature_dim)
        optimizer = torch.optim.Adam([Bi, Gu, Gi, Tu, E, Bp], lr=self.learning_rate)

        for epoch in range(1, self.n_epochs + 1):
            pbar = tqdm.tqdm(total=train_set.num_batches(self.batch_size),
                             desc='Epoch {}/{}'.format(epoch, self.n_epochs))
            for batch_u, batch_i, batch_j in train_set.uij_iter(self.batch_size, shuffle=True):
                gamma_u = Gu[batch_u]
                theta_u = Tu[batch_u]
                beta_prime = Bp[batch_u]

                beta_i = Bi[batch_i]
                beta_j = Bi[batch_j]
                gamma_i = Gi[batch_i]
                gamma_j = Gi[batch_j]
                feature_i = F[batch_i]
                feature_j = F[batch_j]

                Xui = beta_i \
                      + torch.sum(gamma_u * gamma_i, dim=1) \
                      + torch.sum(theta_u * (feature_i.mm(E)), dim=1) \
                      + torch.sum(beta_prime * feature_i, dim=1)

                Xuj = beta_j \
                      + torch.sum(gamma_u * gamma_j, dim=1) \
                      + torch.sum(theta_u * (feature_j.mm(E)), dim=1) \
                      + torch.sum(beta_prime * feature_j, dim=1)

                reg = self.lambda_t * self._l2_loss(theta_u, gamma_u, gamma_i, gamma_j) \
                      + self.lambda_b * self._l2_loss(beta_prime) \
                      + self.lambda_e * self._l2_loss(E)

                loss = - torch.log(torch.sigmoid(Xui - Xuj) + 1e-10).sum() + reg / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
            pbar.close()
            print('Training loss: {:.5f}\n'.format(loss.data.item()))

        self.beta_item = Bi.data.cpu().numpy()
        self.gamma_user = Gu.data.cpu().numpy()
        self.gamma_item = Gi.data.cpu().numpy()
        self.theta_user = Tu.data.cpu().numpy()
        self.emb_matrix = E.data.cpu().numpy()
        self.beta_prime = Bp.cpu().data.numpy()
        self.theta_item = F.mm(E).data.cpu().numpy() # pre-computed projections for faster evaluation

        print('Optimization finished!')

    def rank(self, user_id, candidate_item_ids=None):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_id: int, required
            The index of the user for whom to perform item raking.

        candidate_item_ids: 1d array, optional, default: None
            A list of item indices to be ranked by the user.
            If `None`, list of ranked known item indices will be returned

        Returns
        -------
        Numpy 1d array
            Array of item indices sorted (in decreasing order) relative to some user preference scores.
        """
        known_item_scores = self.beta_item
        if not self.train_set.is_unk_user(user_id):
            known_item_scores += np.dot(self.gamma_item, self.gamma_user[user_id])
            known_item_scores += np.dot(self.theta_item, self.theta_user[user_id])
            known_item_scores += np.dot(self.item_feature, self.beta_prime[user_id])

        if candidate_item_ids is None:
            ranked_item_ids = known_item_scores.argsort()[::-1]
            return ranked_item_ids
        else:
            num_items = max(self.train_set.num_items, max(candidate_item_ids) + 1)
            pref_scores = np.ones(num_items) * self.train_set.min_rating  # use min_rating to shift unk items to the end
            pref_scores[:self.train_set.num_items] = known_item_scores

            ranked_item_ids = pref_scores.argsort()[::-1]
            ranked_item_ids = intersects(ranked_item_ids, candidate_item_ids, assume_unique=True)

            return ranked_item_ids