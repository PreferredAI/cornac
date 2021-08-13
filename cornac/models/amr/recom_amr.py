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
from tqdm.auto import tqdm

from ..recommender import Recommender
from ...exception import CornacException
from ...exception import ScoreException
from ...utils import fast_dot
from ...utils.common import intersects
from ...utils import get_rng
from ...utils.init_utils import zeros, xavier_uniform


class AMR(Recommender):
    """Adversarial Training Towards Robust Multimedia Recommender System.

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

    lambda_adv: float, optional, default: 1.0
        The regularization hyper-parameter in Eq. (8) and (10) for the adversarial sample loss.

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

    seed: int, optional, default: None
        Random seed for weight initialization.

    References
    ----------
    * Tang, J., Du, X., He, X., Yuan, F., Tian, Q., and Chua, T. (2020). Adversarial Training Towards Robust Multimedia Recommender System.
    """
    
    def __init__(
            self,
            name="AMR",
            k=10,
            k2=10,
            n_epochs=50,
            batch_size=100,
            learning_rate=0.005,
            lambda_w=0.01,
            lambda_b=0.01,
            lambda_e=0.0,
            lambda_adv=1.0,
            use_gpu=False,
            trainable=True,
            verbose=True,
            init_params=None,
            seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.k2 = k2
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_w = lambda_w
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e
        self.lambda_adv = lambda_adv
        self.use_gpu = use_gpu
        self.seed = seed
        
        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.gamma_user = self.init_params.get("Gu", None)
        self.gamma_item = self.init_params.get("Gi", None)
        self.emb_matrix = self.init_params.get("E", None)
    
    def _init(self, n_users, n_items, features):
        rng = get_rng(self.seed)
        
        if self.gamma_user is None:
            self.gamma_user = xavier_uniform((n_users, self.k), rng)
        if self.gamma_item is None:
            self.gamma_item = xavier_uniform((n_items, self.k), rng)
        if self.emb_matrix is None:
            self.emb_matrix = xavier_uniform((features.shape[1], self.k), rng)
        
        # pre-computed for faster evaluation
        self.theta_item = np.matmul(features, self.emb_matrix)
    
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
        
        if train_set.item_image is None:
            raise CornacException("item_image modality is required but None.")
        
        # Item visual feature from CNN
        train_features = train_set.item_image.features[: self.train_set.total_items]
        train_features = train_features.astype(np.float32)
        self._init(
            n_users=train_set.total_users,
            n_items=train_set.total_items,
            features=train_features,
        )
        
        if self.trainable:
            self._fit_torch(train_features)
        
        return self
    
    def _fit_torch(self, train_features):
        import torch
        
        def _l2_loss(*tensors):
            l2_loss = 0
            for tensor in tensors:
                l2_loss += tensor.pow(2).sum()
            return l2_loss / 2
        
        def _inner(a, b):
            return (a * b).sum(dim=1)
        
        dtype = torch.float
        device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )
        
        # set requireds_grad=True to get the adversarial gradient
        # if F is not put into the optimization list of parameters
        # it won't be updated
        F = torch.tensor(
            train_features, device=device, dtype=dtype, requires_grad=True
        )
        # Learned parameters
        Gu = torch.tensor(
            self.gamma_user, device=device, dtype=dtype, requires_grad=True
        )
        Gi = torch.tensor(
            self.gamma_item, device=device, dtype=dtype, requires_grad=True
        )
        E = torch.tensor(
            self.emb_matrix, device=device, dtype=dtype, requires_grad=True
        )
        
        optimizer = torch.optim.Adam([Gu, Gi, E], lr=self.learning_rate)
        
        for epoch in range(1, self.n_epochs + 1):
            sum_loss = 0.0
            count = 0
            progress_bar = tqdm(
                total=self.train_set.num_batches(self.batch_size),
                desc="Epoch {}/{}".format(epoch, self.n_epochs),
                disable=not self.verbose,
            )
            for batch_u, batch_i, batch_j in self.train_set.uij_iter(
                    self.batch_size, shuffle=True
            ):
                gamma_u = Gu[batch_u]
                gamma_i = Gi[batch_i]
                gamma_j = Gi[batch_j]
                feat_i = F[batch_i]
                feat_j = F[batch_j]
                
                gamma_diff = gamma_i - gamma_j
                feat_diff = feat_i - feat_j
                
                Xuij = (
                        _inner(gamma_u, gamma_diff)
                        + _inner(gamma_u, feat_diff.mm(E))
                )
                
                log_likelihood = torch.nn.functional.logsigmoid(Xuij).sum()
                
                # adversarial part
                feat_i.retain_grad()
                feat_j.retain_grad()
                log_likelihood.backward(retain_graph=True)
                feat_i_delta = feat_i.grad
                feat_j_delta = feat_j.grad
                
                adv_feat_diff = feat_diff + (feat_i_delta - feat_j_delta)
                adv_Xuij = (
                        _inner(gamma_u, gamma_diff)
                        + _inner(gamma_u, adv_feat_diff.mm(E))
                )
                
                adv_log_likelihood = torch.nn.functional.logsigmoid(adv_Xuij).sum()
                
                reg = (
                        _l2_loss(gamma_u, gamma_i, gamma_j) * self.lambda_w
                        + _l2_loss(E) * self.lambda_e
                )
                
                loss = -log_likelihood - self.lambda_adv * adv_log_likelihood + reg
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                sum_loss += loss.data.item()
                count += len(batch_u)
                if count % (self.batch_size * 10) == 0:
                    progress_bar.set_postfix(loss=(sum_loss / count))
                progress_bar.update(1)
            progress_bar.close()
        
        print("Optimization finished!")
        
        self.gamma_user = Gu.data.cpu().numpy()
        self.gamma_item = Gi.data.cpu().numpy()
        self.emb_matrix = E.data.cpu().numpy()
        # pre-computed for faster evaluation
        self.theta_item = F.mm(E).data.cpu().numpy()
    
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
            known_item_scores = np.zeros(self.gamma_item.shape[0], dtype=np.float32)
            fast_dot(self.gamma_user[user_idx], self.gamma_item, known_item_scores)
            fast_dot(self.gamma_user[user_idx], self.theta_item, known_item_scores)
            return known_item_scores
        else:
            item_score = np.dot(self.gamma_item[item_idx], self.gamma_user[user_idx])
            item_score += np.dot(self.theta_item[item_idx], self.gamma_user[user_idx])
            return item_score
