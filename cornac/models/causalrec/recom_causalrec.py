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
from scipy.special import expit

from ..recommender import Recommender
from ...exception import CornacException
from ...exception import ScoreException
from ...utils import fast_dot
from ...utils.common import intersects
from ...utils import get_rng
from ...utils.init_utils import zeros, xavier_uniform


class CausalRec(Recommender):
    """CausalRec: Causal Inference for Visual Debiasing in Visually-Aware Recommendation

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
    
    mean_feat: torch.tensor, required, default: None
        The mean feature of all item embeddings serving as the no-treatment during causal inference.
    
    tanh: int, optional, default: 0
        The number of tanh layers on the visual feature transformation.
    
    lambda_2: float, optional, default: 0.8
        The coefficient controlling the elimination of the visual bias in Eq. (28).

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
    * Qiu R., Wang S., Chen Z., Yin H., Huang Z. (2021). CausalRec: Causal Inference for Visual Debiasing in Visually-Aware Recommendation.
    """
    
    def __init__(
            self,
            name="CausalRec",
            k=10,
            k2=10,
            n_epochs=50,
            batch_size=100,
            learning_rate=0.005,
            lambda_w=0.01,
            lambda_b=0.01,
            lambda_e=0.0,
            mean_feat=None,
            tanh=0,
            lambda_2=0.8,
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
        self.mean_feat = mean_feat
        self.tanh = tanh
        self.lambda_2 = lambda_2
        self.use_gpu = use_gpu
        self.seed = seed
        
        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.beta_item = self.init_params.get("Bi", None)
        self.gamma_user = self.init_params.get("Gu", None)
        self.gamma_item = self.init_params.get("Gi", None)
        self.theta_user = self.init_params.get("Tu", None)
        self.emb_matrix = self.init_params.get("E", None)
        self.emb_ind_matrix = self.init_params.get("E_ind", None)
        if self.tanh == 2:
            self.emb_matrix2 = self.init_params.get("E2", None)
            self.emb_ind_matrix2 = self.init_params.get("E_ind2", None)
        self.beta_prime = self.init_params.get("Bp", None)
    
    def _init(self, n_users, n_items, features):
        rng = get_rng(self.seed)
        
        self.beta_item = zeros(n_items) if self.beta_item is None else self.beta_item
        if self.gamma_user is None:
            self.gamma_user = xavier_uniform((n_users, self.k), rng)
        if self.gamma_item is None:
            self.gamma_item = xavier_uniform((n_items, self.k), rng)
        if self.theta_user is None:
            self.theta_user = xavier_uniform((n_users, self.k2), rng)
        if self.emb_matrix is None:
            self.emb_matrix = xavier_uniform((features.shape[1], self.k2), rng)
        if self.emb_ind_matrix is None:
            self.emb_ind_matrix = xavier_uniform((features.shape[1], self.k), rng)
        if self.tanh == 2:
            if self.emb_matrix2 is None:
                self.emb_matrix2 = xavier_uniform((self.k2, self.k2), rng)
            if self.emb_ind_matrix2 is None:
                self.emb_ind_matrix2 = xavier_uniform((self.k, self.k), rng)
        if self.beta_prime is None:
            self.beta_prime = xavier_uniform((features.shape[1], 1), rng)
        
        # pre-computed for faster evaluation
        self.theta_item = np.matmul(features, self.emb_matrix)
        self.visual_bias = np.matmul(features, self.beta_prime).ravel()
        self.direct_theta_item = np.matmul(features, self.emb_ind_matrix)
    
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
        
        F = torch.tensor(train_features, device=device, dtype=dtype)
        # Learned parameters
        Bi = torch.tensor(
            self.beta_item, device=device, dtype=dtype, requires_grad=True
        )
        Gu = torch.tensor(
            self.gamma_user, device=device, dtype=dtype, requires_grad=True
        )
        Gi = torch.tensor(
            self.gamma_item, device=device, dtype=dtype, requires_grad=True
        )
        Tu = torch.tensor(
            self.theta_user, device=device, dtype=dtype, requires_grad=True
        )
        E = torch.tensor(
            self.emb_matrix, device=device, dtype=dtype, requires_grad=True
        )
        Bp = torch.tensor(
            self.beta_prime, device=device, dtype=dtype, requires_grad=True
        )
        E_ind = torch.tensor(
            self.emb_ind_matrix, device=device, dtype=dtype, requires_grad=True
        )
        mean_feat = torch.tensor(
            [self.mean_feat], device=device, dtype=dtype, requires_grad=False
        )
        param = [Bi, Gu, Gi, Tu, E, Bp, E_ind]
        
        if self.tanh == 2:
            E2 = torch.tensor(
                self.emb_matrix2, device=device, dtype=dtype, requires_grad=True
            )
            E_ind2 = torch.tensor(
                self.emb_ind_matrix2, device=device, dtype=dtype, requires_grad=True
            )
            param.append(E2)
            param.append(E_ind2)
        
        optimizer = torch.optim.Adam(param, lr=self.learning_rate)
        
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
                theta_u = Tu[batch_u]
                
                beta_i = Bi[batch_i]
                beta_j = Bi[batch_j]
                gamma_i = Gi[batch_i]
                gamma_j = Gi[batch_j]
                feat_i = F[batch_i]
                feat_j = F[batch_j]
                
                if self.tanh == 0:
                    direct_feat_i = feat_i.mm(E)
                    ind_feat_i = feat_i.mm(E_ind)
                elif self.tanh == 1:
                    direct_feat_i = torch.tanh(feat_i.mm(E))
                    ind_feat_i = torch.tanh(feat_i.mm(E_ind))
                elif self.tanh == 2:
                    direct_feat_i = torch.tanh(torch.tanh(feat_i.mm(E)).mm(E2))
                    ind_feat_i = torch.tanh(torch.tanh(feat_i.mm(E_ind)).mm(E_ind2))
                
                i_m = beta_i + _inner(gamma_u, gamma_i) + _inner(gamma_u, gamma_i * ind_feat_i)
                i_n = _inner(theta_u, direct_feat_i) + feat_i.mm(Bp)
                
                if self.tanh == 0:
                    direct_feat_j = feat_j.mm(E)
                    ind_feat_j = feat_j.mm(E_ind)
                elif self.tanh == 1:
                    direct_feat_j = torch.tanh(feat_j.mm(E))
                    ind_feat_j = torch.tanh(feat_j.mm(E_ind))
                elif self.tanh == 2:
                    direct_feat_j = torch.tanh(torch.tanh(feat_j.mm(E)).mm(E2))
                    ind_feat_j = torch.tanh(torch.tanh(feat_j.mm(E_ind)).mm(E_ind2))
                
                j_m = beta_j + _inner(gamma_u, gamma_j) + _inner(gamma_u, gamma_j * ind_feat_j)
                j_n = _inner(theta_u, direct_feat_j) + feat_j.mm(Bp)
                
                i_score = torch.sigmoid(i_m + i_n) * torch.sigmoid(i_m) * torch.sigmoid(i_n)
                j_score = torch.sigmoid(j_m + j_n) * torch.sigmoid(j_m) * torch.sigmoid(j_n)
                
                log_likelihood = torch.nn.functional.logsigmoid(i_score - j_score).sum()
                log_likelihood_m = torch.nn.functional.logsigmoid(i_m - j_m).sum()
                log_likelihood_n = torch.nn.functional.logsigmoid(i_n - j_n).sum()
                
                if self.tanh < 2:
                    l2_e = _l2_loss(E, Bp, E_ind)
                else:
                    l2_e = _l2_loss(E, Bp, E_ind, E2, E_ind2)
                
                reg = (
                        _l2_loss(gamma_u, gamma_i, gamma_j, theta_u) * self.lambda_w
                        + _l2_loss(beta_i) * self.lambda_b
                        + _l2_loss(beta_j) * self.lambda_b / 10
                        + l2_e * self.lambda_e
                )
                
                loss = -log_likelihood + reg - log_likelihood_m - log_likelihood_n
                
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
        
        self.beta_item = Bi.data.cpu().numpy()
        self.gamma_user = Gu.data.cpu().numpy()
        self.gamma_item = Gi.data.cpu().numpy()
        self.theta_user = Tu.data.cpu().numpy()
        self.emb_matrix = E.data.cpu().numpy()
        self.beta_prime = Bp.data.cpu().numpy()
        self.emb_dir_matrix = E_ind.data.cpu().numpy()
        # pre-computed for faster evaluation
        self.theta_item = F.mm(E)
        if self.tanh == 0:
            self.theta_item = self.theta_item.data.cpu().numpy()
        elif self.tanh == 1:
            self.theta_item = torch.tanh(self.theta_item).data.cpu().numpy()
        elif self.tanh == 2:
            self.theta_item = torch.tanh(torch.tanh(self.theta_item).mm(E2)).data.cpu().numpy()

        self.visual_bias = F.mm(Bp).squeeze().data.cpu().numpy()

        self.ind_theta_item = F.mm(E_ind)
        if self.tanh == 0:
            self.ind_theta_item = self.ind_theta_item.data.cpu().numpy()
        elif self.tanh == 1:
            self.ind_theta_item = torch.tanh(self.ind_theta_item).data.cpu().numpy()
        elif self.tanh == 2:
            self.ind_theta_item = torch.tanh(torch.tanh(self.ind_theta_item).mm(E_ind2)).data.cpu().numpy()

        self.beta_item_mean = Bi.mean().unsqueeze(dim=0).data.cpu().numpy()
        self.gamma_item_mean = Gi.mean(dim=0).unsqueeze(dim=0).data.cpu().numpy()

        self.mean_feat = mean_feat.mm(E_ind)
        if self.tanh == 0:
            self.mean_feat = self.mean_feat.data.cpu().numpy()
        elif self.tanh == 1:
            self.mean_feat = torch.tanh(self.mean_feat).data.cpu().numpy()
        elif self.tanh == 2:
            self.mean_feat = torch.tanh(torch.tanh(self.mean_feat).mm(E_ind2)).data.cpu().numpy()

    def score(self, user_idx, item_idx=None):
        """Predict the debiased scores/ratings of a user for an item.

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
            m_score = self.beta_item
            fast_dot(self.gamma_user[user_idx], self.gamma_item, m_score)
            fast_dot(self.gamma_user[user_idx], self.gamma_item * self.ind_theta_item, m_score)

            m_star = self.beta_item_mean
            fast_dot(self.gamma_user[user_idx], self.gamma_item_mean, m_star)
            fast_dot(self.gamma_user[user_idx], self.gamma_item_mean * self.mean_feat, m_star)
            
            n_score = self.visual_bias
            fast_dot(self.theta_user[user_idx], self.theta_item, n_score)

            return expit(m_score + n_score) * expit(m_score) * expit(n_score)\
                   - self.lambda_2 * expit(m_star + n_score) * expit(m_star) * expit(n_score)
        else:
            raise NotImplementedError("The sampled evaluation is not implemented!")
