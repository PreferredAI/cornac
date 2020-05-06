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
from tqdm.auto import trange

from ...utils import estimate_batches

torch.set_default_dtype(torch.float32)

EPS = 1e-10

ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}


class VAE(nn.Module):
    def __init__(self, z_dim, ae_structure, act_fn, likelihood):
        super(VAE, self).__init__()

        self.likelihood = likelihood
        self.act_fn = ACT.get(act_fn, None)
        if self.act_fn is None:
            raise ValueError("Supported act_fn: {}".format(ACT.keys()))

        # Encoder
        self.encoder = nn.Sequential()
        for i in range(len(ae_structure) - 1):
            self.encoder.add_module(
                "fc{}".format(i), nn.Linear(ae_structure[i], ae_structure[i + 1])
            )
            self.encoder.add_module("act{}".format(i), self.act_fn)
        self.enc_mu = nn.Linear(ae_structure[-1], z_dim)  # mu
        self.enc_logvar = nn.Linear(ae_structure[-1], z_dim)  # logvar

        # Decoder
        ae_structure = [z_dim] + ae_structure[::-1]
        self.decoder = nn.Sequential()
        for i in range(len(ae_structure) - 1):
            self.decoder.add_module(
                "fc{}".format(i), nn.Linear(ae_structure[i], ae_structure[i + 1])
            )
            if i != len(ae_structure) - 2:
                self.decoder.add_module("act{}".format(i), self.act_fn)

    def encode(self, x):
        h = self.encoder(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, z):
        h = self.decoder(z)
        if self.likelihood == "mult":
            return torch.softmax(h, dim=1)
        else:
            return torch.sigmoid(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, x_, mu, logvar, beta):
        # Likelihood
        ll_choices = {
            "mult": x * torch.log(x_ + EPS),
            "bern": x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS),
            "gaus": -(x - x_) ** 2,
            "pois": x * torch.log(x_ + EPS) - x_,
        }

        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Supported likelihoods: {}".format(ll_choices.keys()))

        ll = torch.sum(ll, dim=1)

        # KL term
        std = torch.exp(0.5 * logvar)
        kld = -0.5 * (1 + 2.0 * torch.log(std) - mu.pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(beta * kld - ll)


def learn(
    vae,
    train_set,
    n_epochs,
    batch_size,
    learn_rate,
    beta,
    verbose,
    device=torch.device("cpu"),
):
    optimizer = torch.optim.Adam(params=vae.parameters(), lr=learn_rate)
    num_steps = estimate_batches(train_set.num_users, batch_size)

    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    for _ in progress_bar:
        sum_loss = 0.0
        count = 0
        for batch_id, u_ids in enumerate(
            train_set.user_iter(batch_size, shuffle=False)
        ):
            u_batch = train_set.matrix[u_ids, :]
            u_batch.data = np.ones(len(u_batch.data))  # Binarize data
            u_batch = u_batch.A
            u_batch = torch.tensor(u_batch, dtype=torch.float32, device=device)

            # Reconstructed batch
            u_batch_, mu, logvar = vae(u_batch)

            loss = vae.loss(u_batch, u_batch_, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(u_batch)

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))

    return vae
