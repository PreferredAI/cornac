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
from tqdm.auto import tqdm


def ibpr(
    train_set,
    k,
    lamda=0.001,
    n_epochs=150,
    learning_rate=0.05,
    batch_size=100,
    init_params=None,
    verbose=False,
):
    X = train_set.csr_matrix

    # Initial user factors
    if init_params["U"] is None:
        U = torch.randn(X.shape[0], k, requires_grad=True)
    else:
        U = torch.from_numpy(init_params["U"])
        U.requires_grad = True

    # Initial item factors
    if init_params["V"] is None:
        V = torch.randn(X.shape[1], k, requires_grad=True)
    else:
        V = torch.from_numpy(init_params["V"])
        V.requires_grad = True
        
    optimizer = torch.optim.Adam([U, V], lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        sum_loss = 0.0
        count = 0
        progress_bar = tqdm(
            total=train_set.num_batches(batch_size),
            desc="Epoch {}/{}".format(epoch, n_epochs),
            disable=not verbose,
        )

        for batch_u, batch_i, batch_j in train_set.uij_iter(batch_size, shuffle=True):
            regU = U[batch_u, :]
            regI = V[batch_i, :]
            regJ = V[batch_j, :]

            regU_unq = U[np.unique(batch_u), :]
            regI_unq = V[np.union1d(batch_i, batch_j), :]

            regU_norm = regU / regU.norm(dim=1)[:, None]
            regI_norm = regI / regI.norm(dim=1)[:, None]
            regJ_norm = regJ / regJ.norm(dim=1)[:, None]

            Scorei = torch.acos(
                torch.clamp(
                    torch.sum(regU_norm * regI_norm, dim=1), -1 + 1e-7, 1 - 1e-7
                )
            )
            Scorej = torch.acos(
                torch.clamp(
                    torch.sum(regU_norm * regJ_norm, dim=1), -1 + 1e-7, 1 - 1e-7
                )
            )

            loss = (
                lamda * (regU_unq.norm().pow(2) + regI_unq.norm().pow(2))
                - torch.log(torch.sigmoid(Scorej - Scorei)).sum()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(batch_u)
            if count % (batch_size * 10) == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))
            progress_bar.update(1)

        progress_bar.close()

    # since the user's preference is defined by the angular distance,
    # we can normalize the user/item vectors without changing the ranking
    U = torch.nn.functional.normalize(U, p=2, dim=1)
    V = torch.nn.functional.normalize(V, p=2, dim=1)
    U = U.data.cpu().numpy()
    V = V.data.cpu().numpy()

    res = {"U": U, "V": V}

    return res
