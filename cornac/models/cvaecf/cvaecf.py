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


class CVAE(nn.Module):
    def __init__(self, z_dim, h_dim, ae_structure_z, ae_structure_h, act_fn, likelihood):
        super(CVAE, self).__init__()

        self.likelihood = likelihood
        self.act_fn = ACT.get(act_fn, None)
        if self.act_fn is None:
            raise ValueError("Supported act_fn: {}".format(ACT.keys()))

        # Encoder for preference data y, q(z|y)
        self.encoder_qz = nn.Sequential()
        for i in range(len(ae_structure_z) - 1):
            self.encoder_qz.add_module(
                "fc{}".format(i), nn.Linear(ae_structure_z[i], ae_structure_z[i + 1])
            )
            self.encoder_qz.add_module("act{}".format(i), self.act_fn)
        self.enc_qz_mu = nn.Linear(ae_structure_z[-1], z_dim)  # mu
        self.enc_qz_logvar = nn.Linear(ae_structure_z[-1], z_dim)  # logvar

        # Encoder for auxiliary data x, q(h|x)
        self.encoder_qhx = nn.Sequential()
        for i in range(len(ae_structure_h) - 1):
            self.encoder_qhx.add_module(
                "fc{}".format(i), nn.Linear(ae_structure_h[i], ae_structure_h[i + 1])
            )
            self.encoder_qhx.add_module("act{}".format(i), self.act_fn)
        self.enc_qhx_mu = nn.Linear(ae_structure_h[-1], h_dim)  # mu
        self.enc_qhx_logvar = nn.Linear(ae_structure_h[-1], h_dim)  # logvar

        # Encoder for preference data y, q(h|y)
        self.encoder_qhy = nn.Sequential()
        for i in range(len(ae_structure_z) - 1):
            self.encoder_qhy.add_module(
                "fc{}".format(i), nn.Linear(ae_structure_z[i], ae_structure_z[i + 1])
            )
            self.encoder_qhy.add_module("act{}".format(i), self.act_fn)
        self.enc_qhy_mu = nn.Linear(ae_structure_z[-1], h_dim)  # mu
        self.enc_qhy_logvar = nn.Linear(ae_structure_z[-1], h_dim)  # logvar

        # Encoder for auxiliary data x, p(h|x)
        self.encoder_phx = nn.Sequential()
        for i in range(len(ae_structure_h) - 1):
            self.encoder_phx.add_module(
                "fc{}".format(i), nn.Linear(ae_structure_h[i], ae_structure_h[i + 1])
            )
            self.encoder_phx.add_module("act{}".format(i), self.act_fn)
        self.enc_ph_mu = nn.Linear(ae_structure_h[-1], h_dim)  # mu
        self.enc_ph_logvar = nn.Linear(ae_structure_h[-1], h_dim)  # logvar

        # Decoder for preference data x
        ae_structure_z = [z_dim + h_dim] + ae_structure_z[::-1]
        self.decoder = nn.Sequential()
        for i in range(len(ae_structure_z) - 1):
            self.decoder.add_module(
                "fc{}".format(i), nn.Linear(ae_structure_z[i], ae_structure_z[i + 1])
            )
            if i != len(ae_structure_z) - 2:
                self.decoder.add_module("act{}".format(i), self.act_fn)

    def encode_qz(self, y):
        o = self.encoder_qz(y)
        return self.enc_qz_mu(o), self.enc_qz_logvar(o)

    def encode_qhx(self, x):
        o = self.encoder_qhx(x)
        return self.enc_qhx_mu(o), self.enc_qhx_logvar(o)

    def encode_qhy(self, y):
        o = self.encoder_qhy(y)
        return self.enc_qhy_mu(o), self.enc_qhy_logvar(o)

    def encode_phx(self, x):
        o = self.encoder_phx(x)
        return self.enc_ph_mu(o), self.enc_ph_logvar(o)

    def decode(self, z, h):
        zh = torch.cat([z, h], 1)
        h = self.decoder(zh)
        if self.likelihood == "mult":
            return torch.softmax(h, dim=1)
        else:
            return torch.sigmoid(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, y, x):
        mu_qz, logvar_qz = self.encode_qz(y)
        mu_qhx, logvar_qhx = self.encode_qhx(x)
        mu_qhy, logvar_qhy = self.encode_qhy(y)
        mu_ph, logvar_ph = self.encode_phx(x)

        z = self.reparameterize(mu_qz, logvar_qz)
        h_q = self.reparameterize(mu_qhx, logvar_qhx)

        return self.decode(z, h_q), mu_qz, logvar_qz, mu_qhx, logvar_qhx, mu_qhy, logvar_qhy, mu_ph, logvar_ph

    def loss(self, x, x_, mu_qz, logvar_qz, mu_qhx, logvar_qhx, mu_qhy, logvar_qhy, mu_ph, logvar_ph, beta, alpha_1,
             alpha_2):
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

        # KL term z
        std_qz = torch.exp(0.5 * logvar_qz)
        kld_z = -0.5 * (1 + 2.0 * torch.log(std_qz) - mu_qz.pow(2) - std_qz.pow(2))
        kld_z = torch.sum(kld_z, dim=1)

        # KL term h
        std_qhx = torch.exp(0.5 * logvar_qhx)
        std_qhy = torch.exp(0.5 * logvar_qhy)
        std_ph = torch.exp(0.5 * logvar_ph)

        # KL(q(h|x)||p(h|x))
        kld_hx = -0.5 * (1 + 2.0 * torch.log(std_qhx) - (mu_qhx - mu_ph).pow(2) - std_qhx.pow(
            2))  # assuming std_ph is 1 for now
        kld_hx = torch.sum(kld_hx, dim=1)

        # KL(q(h|x)||q(h|y))
        kld_hy = -0.5 * (1 + 2.0 * torch.log(std_qhx) - 2.0 * torch.log(std_qhy) - (
                    (mu_qhx - mu_qhy).pow(2) + std_qhx.pow(2)) / std_qhy.pow(2))  # assuming std_ph is 1 for now
        kld_hy = torch.sum(kld_hy, dim=1)

        return torch.mean(beta * kld_z + alpha_1 * kld_hx + alpha_2 * kld_hy - ll)


def learn(
        cvae,
        train_set,
        n_epochs,
        batch_size,
        learn_rate,
        beta,
        alpha_1,
        alpha_2,
        verbose,
        device=torch.device("cpu"),
):
    optimizer = torch.optim.Adam(params=cvae.parameters(), lr=learn_rate)

    x = train_set.user_graph.matrix[: train_set.num_users, : train_set.num_users]
    y = train_set.matrix
    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    for _ in progress_bar:
        sum_loss = 0.0
        count = 0
        for batch_id, u_ids in enumerate(
                train_set.user_iter(batch_size, shuffle=False)
        ):
            y_batch = y[u_ids, :]
            y_batch.data = np.ones(len(y_batch.data))  # Binarize data
            y_batch = y_batch.A
            y_batch = torch.tensor(y_batch, dtype=torch.float32, device=device)

            x_batch = x[u_ids, :]
            x_batch = x_batch.A
            x_batch = torch.tensor(x_batch, dtype=torch.float32, device=device)

            # Reconstructed batch
            y_batch_, mu_qz, logvar_qz, mu_qhx, logvar_qhx, mu_qhy, logvar_qhy, mu_ph, logvar_ph = cvae(y_batch,
                                                                                                        x_batch)

            loss = cvae.loss(y_batch, y_batch_, mu_qz, logvar_qz, mu_qhx, logvar_qhx, mu_qhy, logvar_qhy, mu_ph,
                             logvar_ph, alpha_1, alpha_2, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += y_batch.shape[0]

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))

    return cvae
