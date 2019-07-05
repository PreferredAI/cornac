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


def online_ibpr(triplets, k, lamda=0.005, n_epochs=150, learning_rate=0.001, batch_size=100, init_params=None):
    U = init_params['U']
    U = torch.tensor(U, requires_grad=True)

    # Initial item factors
    V = init_params['V']
    V = torch.tensor(V)

    optimizer = torch.optim.Adam([U], lr=learning_rate)
    for epoch in range(n_epochs):
        regU = U[triplets[:, 0], :]
        regI = V[triplets[:, 1], :]
        regJ = V[triplets[:, 2], :]

        regU_unq = U[np.unique(triplets[:, 0]), :]
        regI_unq = V[np.unique(triplets[:, 1:]), :]

        regU_norm = regU / regU.norm(dim=1)[:, None]
        regI_norm = regI / regI.norm(dim=1)[:, None]
        regJ_norm = regJ / regJ.norm(dim=1)[:, None]

        Scorei = torch.acos(torch.clamp(torch.sum(regU_norm * regI_norm, dim=1), -1 + 1e-7, 1 - 1e-7))
        Scorej = torch.acos(torch.clamp(torch.sum(regU_norm * regJ_norm, dim=1), -1 + 1e-7, 1 - 1e-7))

        loss = lamda * (regU_unq.norm().pow(2) + regI_unq.norm().pow(2)) - torch.log(
            torch.sigmoid(Scorej - Scorei)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch:', epoch, 'loss:', loss)

    # since the user's preference is defined by the angular distance, we can normalize the user/item vectors without changing the ranking
    # U = torch.nn.functional.normalize(U, p = 2, dim=1)
    # V = torch.nn.functional.normalize(V, p = 2, dim=1)
    U = U.data.cpu().numpy()
    V = V.data.cpu().numpy()

    res = {'U': U, 'V': V}

    return res
