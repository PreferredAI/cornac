# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao <jyguo@smu.edu.sg>
         Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..recommender import Recommender
from ...exception import CornacException
import numpy as np
from tqdm import tqdm
from ...utils import fast_dot

try:
    import torch
except ImportError:
    torch = None


class VBPR(Recommender):
    """Visual Bayesian Personalized Ranking.

    Parameters
    ----------
    k: int, optional, default: 10
        The dimension of the gamma latent factors.

    k2: int, optional, default: 10
        The dimension of the theta latent factors.

    n_epochs: int, optional, default: 20
        Maximum number of epochs for SGD.

    batch_size: int, optional, default: 100
        The batch size for SGD.

    learning_rate: float, optional, default: 0.001
        The learning rate for SGD.

    lambda_w: float, optional, default: 0.01
        The regularization hyper-parameter for latent factor weights.

    lambda_b: float, optional, default: 0.01
        The regularization hyper-parameter for biases.

    lambda_e: float, optional, default: 0.0
        The regularization hyper-parameter for embedding matrix E and beta prime vector.

    use_gpu: boolean, optional, default: True
        Whether or not to use GPU to speed up training.

    trainable: boolean, optional, default: True
        When False, the model is not trained and Cornac assumes that the model already \
        pre-trained (U and V are not None).

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params = {'Bi': beta_item, 'Gu': gamma_user,
        'Gi': gamma_item, 'Tu': theta_user, 'E': emb_matrix, 'Bp': beta_prime}

    References
    ----------
    * HE, Ruining et MCAULEY, Julian. VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. In : AAAI. 2016. p. 144-150.
    """

    def __init__(self,
                 k=10, k2=10,
                 n_epochs=20, batch_size=100, learning_rate=0.001,
                 lambda_w=0.01, lambda_b=0.01, lambda_e=0.0,
                 use_gpu=False, trainable=True, verbose=True, init_params=None):
        Recommender.__init__(self, name='VBPR', trainable=trainable, verbose=verbose)
        self.k = k
        self.k2 = k2
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_w = lambda_w
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e
        self.init_params = {} if init_params is None else init_params

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

    def _init_params(self, n_users, n_items, feat_dim):
        Bi = self._load_or_randn((n_items), init_values=self.init_params.get('Bi', None))
        Gu = self._load_or_randn((n_users, self.k), init_values=self.init_params.get('Gu', None))
        Gi = self._load_or_randn((n_items, self.k), init_values=self.init_params.get('Gi', None))
        Tu = self._load_or_randn((n_users, self.k2), init_values=self.init_params.get('Tu', None))
        E = self._load_or_randn((feat_dim, self.k2), init_values=self.init_params.get('E', None))
        Bp = self._load_or_randn((feat_dim, 1), init_values=self.init_params.get('Bp', None))

        return Bi, Gu, Gi, Tu, E, Bp

    def _l2_loss(self, *tensors):
        l2_loss = 0
        for tensor in tensors:
            l2_loss += torch.sum(tensor ** 2) / 2
        return l2_loss

    def _inner(self, a, b):
        return torch.sum(a * b, dim=1)

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

        if train_set.item_image is None:
            raise CornacException('item_image module is required but None.')

        # Item visual feature from CNN
        self.item_feature = train_set.item_image.features[:self.train_set.num_items]
        F = torch.from_numpy(self.item_feature).float().to(self.device)

        # Learned parameters
        Bi, Gu, Gi, Tu, E, Bp = self._init_params(n_users=train_set.num_users,
                                                  n_items=train_set.num_items,
                                                  feat_dim=train_set.item_image.feature_dim)
        optimizer = torch.optim.Adam([Bi, Gu, Gi, Tu, E, Bp], lr=self.learning_rate)

        for epoch in range(1, self.n_epochs + 1):
            sum_loss = 0.
            count = 0
            progress_bar = tqdm(total=train_set.num_batches(self.batch_size),
                                desc='Epoch {}/{}'.format(epoch, self.n_epochs),
                                disable=not self.verbose)
            for batch_u, batch_i, batch_j in train_set.uij_iter(self.batch_size, shuffle=True):
                gamma_u = Gu[batch_u]
                theta_u = Tu[batch_u]

                beta_i = Bi[batch_i]
                beta_j = Bi[batch_j]
                gamma_i = Gi[batch_i]
                gamma_j = Gi[batch_j]
                feat_i = F[batch_i]
                feat_j = F[batch_j]

                gamma_diff = gamma_i - gamma_j
                feat_diff = feat_i - feat_j

                Xuij = beta_i - beta_j \
                       + self._inner(gamma_u, gamma_diff) \
                       + self._inner(theta_u, feat_diff.mm(E)) \
                       + feat_diff.mm(Bp)

                log_likelihood = torch.log(torch.sigmoid(Xuij) + 1e-10).sum()

                reg = self.lambda_w * self._l2_loss(gamma_u, gamma_i, gamma_j, theta_u) \
                      + self.lambda_b * self._l2_loss(beta_i) \
                      + self.lambda_b / 10 * self._l2_loss(beta_j) \
                      + self.lambda_e * self._l2_loss(E, Bp)

                loss = - log_likelihood + reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_loss += loss.data.item()
                count += len(batch_u)
                if count % (self.batch_size * 10) == 0:
                    progress_bar.set_postfix(loss=(sum_loss / count))
                progress_bar.update(1)
            progress_bar.close()

        self.beta_item = Bi.data.cpu().numpy()
        self.gamma_user = Gu.data.cpu().numpy()
        self.gamma_item = Gi.data.cpu().numpy()
        self.theta_user = Tu.data.cpu().numpy()
        self.emb_matrix = E.data.cpu().numpy()
        # pre-computed for faster evaluation
        self.theta_item = F.mm(E).data.cpu().numpy()
        self.visual_bias = F.mm(Bp).data.cpu().numpy().ravel()

        print('Optimization finished!')

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
            known_item_scores = np.add(self.beta_item, self.visual_bias)
            if not self.train_set.is_unk_user(user_id):
                fast_dot(self.gamma_user[user_id], self.gamma_item, known_item_scores)
                fast_dot(self.theta_user[user_id], self.theta_item, known_item_scores)
            return known_item_scores
        else:
            item_score = np.add(self.beta_item[item_id], self.visual_bias[item_id])
            if not self.train_set.is_unk_user(user_id):
                item_score += np.dot(self.gamma_item[item_id], self.gamma_user[user_id])
                item_score += np.dot(self.theta_item[item_id], self.theta_user[user_id])
            return item_score
