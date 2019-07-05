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
import torch.nn.functional as F
from tqdm import tqdm

from ...utils.data_utils import Dataset


class VAE(nn.Module):
    def __init__(self, data_dim, z_dim, h_dim):
        super(VAE, self).__init__()

        # Hyperparameters
        self.eps = 1e-10

        # Encoder layers
        self.efc1 = nn.Linear(data_dim, h_dim)
        self.efc21 = nn.Linear(h_dim, z_dim)  # mu
        self.efc22 = nn.Linear(h_dim, z_dim)  # logvar

        # Decoder layers
        self.dfc1 = nn.Linear(z_dim, h_dim)
        self.dfc2 = nn.Linear(h_dim, data_dim)
        # self.efc22 = nn.Linear(h_dim, z_dim) # logvar

    def encode(self, x):
        h = self.efc1(x)
        h = F.relu(h)
        return self.efc21(h), self.efc22(h)

    def decode(self, z):
        h = self.dfc1(z)
        h = F.relu(h)
        o = self.dfc2(h)
        return F.softmax(o)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, x_, mu, logvar, beta):
        # Multinomiale likelihood
        mult_ll = x * torch.log(x_ + self.eps)

        # Poisson log-likelihood
        # poiss_ll = x * torch.log(x_ + self.eps) - x_

        # Bernoulli log-likelihood
        # bern_ll = -x * torch.log(x_ + self.eps) -  0.5*(1-x) * torch.log(1 - x_ + self.eps)

        # Gaussian log-likelihood
        # gauss_ll = (x - x_)**2

        ll = mult_ll
        ll = torch.sum(ll, dim=1)

        # KL term
        std = torch.exp(0.5 * logvar)
        kld = -0.5 * (1 + 2. * torch.log(std) - mu.pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(beta * kld - ll)


def learn(train_set, k, h, n_epochs, batch_size, learn_rate, beta, gamma, verbose, seed, use_gpu):
    if seed is not None:
        torch.manual_seed(seed)

    # Instantiations
    x = Dataset(train_set.matrix)
    data_dim = x.data.shape[1]
    vae = VAE(data_dim, k, h)
    params = list(vae.parameters())

    # optimizer = torch.optim.RMSprop(params=params, lr=learn_rate, alpha=gamma)
    optimizer = torch.optim.Adam(params=params, lr=learn_rate)

    for epoch in range(1, n_epochs + 1):
        sum_loss = 0.
        count = 0
        num_steps = int(x.data.shape[0] / batch_size)

        if verbose:
            progress_bar = tqdm(total=num_steps,
                                desc='Epoch {}/{}'.format(epoch, n_epochs),
                                disable=False)

        for i in range(1, num_steps + 1):
            u_batch, u_ids = x.next_batch(batch_size)
            u_batch.data = np.ones(len(u_batch.data))  # Binarize data
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=torch.double)

            # Reconstructed batch
            u_batch_, mu, logvar = vae(u_batch)

            loss = vae.loss(u_batch, u_batch_, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(u_batch)

            if verbose:
                if count % (batch_size * 10) == 0:
                    progress_bar.set_postfix(loss=(sum_loss / count))
                progress_bar.update(1)
        if verbose:
            progress_bar.close()
            print(sum_loss)

    return vae


torch.set_default_dtype(torch.double)
