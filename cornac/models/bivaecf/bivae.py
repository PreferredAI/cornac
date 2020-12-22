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
import torch
import torch.nn as nn
from tqdm.auto import trange

torch.set_default_dtype(torch.float32)

EPS = 1e-10

ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}


class BiVAE(nn.Module):
    def __init__(self, k, user_e_structure, item_e_structure, act_fn, likelihood, cap_priors, feature_dim, batch_size):
        super(BiVAE, self).__init__()

        self.mu_theta = torch.zeros((item_e_structure[0], k))  # n_users*k
        self.mu_beta = torch.zeros((user_e_structure[0], k))  # n_items*k

        # zero mean for standard normal prior
        self.mu_prior = torch.zeros((batch_size, k), requires_grad=False)

        self.theta_ = torch.randn(item_e_structure[0], k) * 0.01
        self.beta_ = torch.randn(user_e_structure[0], k) * 0.01
        torch.nn.init.kaiming_uniform_(self.theta_, a=np.sqrt(5))

        self.likelihood = likelihood
        self.act_fn = ACT.get(act_fn, None)
        if self.act_fn is None:
            raise ValueError("Supported act_fn: {}".format(ACT.keys()))

        self.cap_priors = cap_priors
        if self.cap_priors.get("user", False):
            self.user_prior_encoder = nn.Linear(feature_dim.get("user"), k)
        if self.cap_priors.get("item", False):
            self.item_prior_encoder = nn.Linear(feature_dim.get("item"), k)

        # User Encoder
        self.user_encoder = nn.Sequential()
        for i in range(len(user_e_structure) - 1):
            self.user_encoder.add_module(
                "fc{}".format(i), nn.Linear(user_e_structure[i], user_e_structure[i + 1])
            )
            self.user_encoder.add_module("act{}".format(i), self.act_fn)
        self.user_mu = nn.Linear(user_e_structure[-1], k)  # mu
        self.user_std = nn.Linear(user_e_structure[-1], k)

        # Item Encoder
        self.item_encoder = nn.Sequential()
        for i in range(len(item_e_structure) - 1):
            self.item_encoder.add_module(
                "fc{}".format(i), nn.Linear(item_e_structure[i], item_e_structure[i + 1])
            )
            self.item_encoder.add_module("act{}".format(i), self.act_fn)
        self.item_mu = nn.Linear(item_e_structure[-1], k)  # mu
        self.item_std = nn.Linear(item_e_structure[-1], k)

    def encode_user_prior(self, x):
        h = self.user_prior_encoder(x)
        return h

    def encode_item_prior(self, x):
        h = self.item_prior_encoder(x)
        return h

    def encode_user(self, x):
        h = self.user_encoder(x)
        return self.user_mu(h), torch.sigmoid(self.user_std(h))

    def encode_item(self, x):
        h = self.item_encoder(x)
        return self.item_mu(h), torch.sigmoid(self.item_std(h))

    def decode_user(self, theta, beta):
        h = theta.mm(beta.t())
        return torch.sigmoid(h)

    def decode_item(self, theta, beta):
        h = beta.mm(theta.t())
        return torch.sigmoid(h)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x, user=True, beta=None, theta=None):

        if user:
            mu, std = self.encode_user(x)
            theta = self.reparameterize(mu, std)
            return theta, self.decode_user(theta, beta), mu, std
        else:
            mu, std = self.encode_item(x)
            beta = self.reparameterize(mu, std)
            return beta, self.decode_item(theta, beta), mu, std

    def loss(self, x, x_, mu, mu_prior, std, kl_beta):
        # Likelihood
        ll_choices = {
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -(x - x_) ** 2,
            "pois": x * torch.log(x_ + EPS) - x_,
        }

        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))

        ll = torch.sum(ll, dim=1)

        # KL term
        kld = -0.5 * (1 + 2.0 * torch.log(std) - (mu - mu_prior).pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(kl_beta * kld - ll)


def learn(
        bivae,
        train_set,
        n_epochs,
        batch_size,
        learn_rate,
        beta_kl,
        verbose,
        device=torch.device("cpu"),
):
    user_params = [{'params': bivae.user_encoder.parameters()}, \
                   {'params': bivae.user_mu.parameters()}, \
                   {'params': bivae.user_std.parameters()}, \
                   ]

    item_params = [{'params': bivae.item_encoder.parameters()}, \
                   {'params': bivae.item_mu.parameters()}, \
                   {'params': bivae.item_std.parameters()}, \
                   ]

    if bivae.cap_priors.get("user", False):
        user_params.append({'params': bivae.user_prior_encoder.parameters()})
        user_features = train_set.user_feature.features[: train_set.num_users]
        user_features = torch.from_numpy(user_features).float().to(device)

    if bivae.cap_priors.get("item", False):
        item_params.append({'params': bivae.item_prior_encoder.parameters()})
        item_features = train_set.item_feature.features[: train_set.num_items]
        item_features = torch.from_numpy(item_features).float().to(device)

    u_optimizer = torch.optim.Adam(params=user_params, lr=learn_rate)
    i_optimizer = torch.optim.Adam(params=item_params, lr=learn_rate)

    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    bivae.beta_ = bivae.beta_.to(device=device)
    bivae.theta_ = bivae.theta_.to(device=device)

    bivae.mu_beta = bivae.mu_beta.to(device=device)
    bivae.mu_theta = bivae.mu_theta.to(device=device)

    bivae.mu_prior = bivae.mu_prior.to(device=device)

    x = train_set.matrix.copy()
    x.data = np.ones(len(x.data))  # Binarize data
    tx = x.transpose()

    for _ in progress_bar:

        # item side
        i_sum_loss = 0.0
        i_count = 0
        for batch_id, i_ids in enumerate(
                train_set.item_iter(batch_size, shuffle=False)
        ):
            i_batch = tx[i_ids, :]
            i_batch = i_batch.A
            i_batch = torch.tensor(i_batch, dtype=torch.float32, device=device)

            # Reconstructed batch
            beta, i_batch_, i_mu, i_std = bivae(i_batch, user=False, theta=bivae.theta_)

            if bivae.cap_priors.get("item", False):
                i_batch_f = item_features[i_ids]
                i_mu_prior = bivae.encode_item_prior(i_batch_f)
            else:
                i_mu_prior = bivae.mu_prior[0:len(i_batch)]

            i_loss = bivae.loss(i_batch, i_batch_, i_mu, i_mu_prior, i_std, beta_kl)
            i_optimizer.zero_grad()
            i_loss.backward()
            i_optimizer.step()

            i_sum_loss += i_loss.data.item()
            i_count += len(i_batch)

            beta, _, i_mu, _ = bivae(i_batch, user=False, theta=bivae.theta_)

            bivae.beta_.data[i_ids] = beta.data
            bivae.mu_beta.data[i_ids] = i_mu.data

        # user side
        u_sum_loss = 0.0
        u_count = 0
        for batch_id, u_ids in enumerate(
                train_set.user_iter(batch_size, shuffle=False)
        ):
            u_batch = x[u_ids, :]
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=torch.float32, device=device)

            # Reconstructed batch
            theta, u_batch_, u_mu, u_std = bivae(u_batch, user=True, beta=bivae.beta_)

            if bivae.cap_priors.get("user", False):
                u_batch_f = user_features[u_ids]
                u_mu_prior = bivae.encode_user_prior(u_batch_f)
            else:
                u_mu_prior = bivae.mu_prior[0:len(u_batch)]

            u_loss = bivae.loss(u_batch, u_batch_, u_mu, u_mu_prior, u_std, beta_kl)
            u_optimizer.zero_grad()
            u_loss.backward()
            u_optimizer.step()

            u_sum_loss += u_loss.data.item()
            u_count += len(u_batch)

            theta, _, u_mu, _ = bivae(u_batch, user=True, beta=bivae.beta_)
            bivae.theta_.data[u_ids] = theta.data
            bivae.mu_theta.data[u_ids] = u_mu.data

            progress_bar.set_postfix(loss_i=(i_sum_loss / i_count), loss_u=(u_sum_loss / (u_count)))

    # infer mu_beta
    for batch_id, i_ids in enumerate(train_set.item_iter(batch_size, shuffle=False)):
        i_batch = tx[i_ids, :]
        i_batch = i_batch.A
        i_batch = torch.tensor(i_batch, dtype=torch.float32, device=device)

        beta, _, i_mu, _ = bivae(i_batch, user=False, theta=bivae.theta_)
        bivae.mu_beta.data[i_ids] = i_mu.data

    # infer mu_theta
    for batch_id, u_ids in enumerate(train_set.user_iter(batch_size, shuffle=False)):
        u_batch = x[u_ids, :]
        u_batch = u_batch.A
        u_batch = torch.tensor(u_batch, dtype=torch.float32, device=device)

        theta, _, u_mu, _ = bivae(u_batch, user=True, beta=bivae.beta_)
        bivae.mu_theta.data[u_ids] = u_mu.data

    return bivae
