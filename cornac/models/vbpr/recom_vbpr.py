# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao <jyguo@smu.edu.sg>
         Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..recommender import Recommender
from ...exception import CornacException
import numpy as np
from ...utils import fast_dot


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

    def __init__(self, name='VBPR', k=10, k2=10,
                 n_epochs=50, batch_size=100, learning_rate=0.005,
                 lambda_w=0.01, lambda_b=0.01, lambda_e=0.0,
                 use_gpu=False, trainable=True, verbose=True,
                 init_params=None):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.k2 = k2
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_w = lambda_w
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e
        self.use_gpu = use_gpu
        self.init_params = {} if init_params is None else init_params

    def _init_factors(self, n_users, n_items, features):
        from ...utils.init_utils import zeros, xavier_uniform

        self.beta_item = self.init_params.get('Bi', zeros(n_items))
        self.gamma_user = self.init_params.get('Gu', xavier_uniform((n_users, self.k)))
        self.gamma_item = self.init_params.get('Gi', xavier_uniform((n_items, self.k)))
        self.theta_user = self.init_params.get('Tu', xavier_uniform((n_users, self.k2)))
        self.emb_matrix = self.init_params.get('E', xavier_uniform((features.shape[1], self.k2)))
        self.beta_prime = self.init_params.get('Bp', xavier_uniform((features.shape[1], 1)))
        # pre-computed for faster evaluation
        self.theta_item = np.matmul(features, self.emb_matrix)
        self.visual_bias = np.matmul(features, self.beta_prime).ravel()

    def _l2_loss(self, *tensors):
        l2_loss = 0
        for tensor in tensors:
            l2_loss += tensor.pow(2).sum()
        return l2_loss / 2

    def _inner(self, a, b):
        return (a * b).sum(dim=1)

    def fit(self, train_set):
        """Fit the model.

        Parameters
        ----------
        train_set: :obj:`cornac.data.MultimodalTrainSet`
            Multimodal training set.

        """
        Recommender.fit(self, train_set)

        if train_set.item_image is None:
            raise CornacException('item_image module is required but None.')

        # Item visual feature from CNN
        train_features = train_set.item_image.features[:self.train_set.num_items].astype(np.float32)
        self._init_factors(n_users=train_set.num_users,
                           n_items=train_set.num_items,
                           features=train_features)

        if not self.trainable:
            return

        import torch
        from tqdm import tqdm

        if self.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        F = torch.from_numpy(train_features).float().to(device)

        # Learned parameters
        Bi = torch.tensor(self.beta_item, dtype=torch.float32, requires_grad=True, device=device)
        Gu = torch.tensor(self.gamma_user, dtype=torch.float32, requires_grad=True, device=device)
        Gi = torch.tensor(self.gamma_item, dtype=torch.float32, requires_grad=True, device=device)
        Tu = torch.tensor(self.theta_user, dtype=torch.float32, requires_grad=True, device=device)
        E = torch.tensor(self.emb_matrix, dtype=torch.float32, requires_grad=True, device=device)
        Bp = torch.tensor(self.beta_prime, dtype=torch.float32, requires_grad=True, device=device)

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

                log_likelihood = torch.nn.functional.logsigmoid(Xuij).sum()

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

        print('Optimization finished!')

        self.beta_item = Bi.data.cpu().numpy()
        self.gamma_user = Gu.data.cpu().numpy()
        self.gamma_item = Gi.data.cpu().numpy()
        self.theta_user = Tu.data.cpu().numpy()
        self.emb_matrix = E.data.cpu().numpy()
        # pre-computed for faster evaluation
        self.theta_item = F.mm(E).data.cpu().numpy()
        self.visual_bias = F.mm(Bp).data.cpu().numpy().ravel()

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
