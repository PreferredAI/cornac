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


from tqdm.auto import tqdm
import numpy as np
import torch

from ...utils.common import scale
from ...utils.init_utils import normal
from ...utils import get_rng

torch.set_default_dtype(torch.double)


def _load_or_randn(size, init_values, seed, device):
    if init_values is None:
        rng = get_rng(seed)
        tensor = normal(size, mean=0.0, std=0.001, random_state=rng, dtype=np.double)
        tensor = torch.tensor(tensor, requires_grad=True, device=device)
    else:
        tensor = torch.tensor(init_values, requires_grad=True, device=device)
    return tensor


def _l2_loss(*tensors):
    l2_loss = 0
    for tensor in tensors:
        l2_loss += torch.sum(tensor ** 2) / 2
    return l2_loss


def vmf(
    train_set,
    item_feature,
    k,
    d,
    n_epochs,
    batch_size,
    lambda_u,
    lambda_v,
    lambda_p,
    lambda_e,
    learning_rate,
    gamma,
    init_params,
    use_gpu,
    verbose,
    seed,
):
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    F = torch.from_numpy(item_feature).double().to(device)

    f_dim = train_set.item_image.feature_dim
    n_users = train_set.num_users
    n_items = train_set.num_items

    # preparing parameters
    U = _load_or_randn(
        (n_users, k), init_values=init_params.get("U"), seed=seed, device=device
    )
    V = _load_or_randn(
        (n_items, k), init_values=init_params.get("V"), seed=seed, device=device
    )
    P = _load_or_randn(
        (n_users, d), init_values=init_params.get("P"), seed=seed, device=device
    )
    E = _load_or_randn(
        (f_dim, d), init_values=init_params.get("E"), seed=seed, device=device
    )

    # optimizer
    optimizer = torch.optim.RMSprop([U, V, P, E], lr=learning_rate, alpha=gamma)

    for epoch in range(1, n_epochs + 1):
        sum_loss = 0.0
        count = 0
        progress_bar = tqdm(
            total=train_set.num_batches(batch_size),
            desc="Epoch {}/{}".format(epoch, n_epochs),
            disable=not verbose,
        )

        for batch_u, batch_i, batch_r in train_set.uir_iter(batch_size, shuffle=True):
            U_u = U[batch_u]
            P_u = P[batch_u]
            V_i = V[batch_i]
            f_i = F[batch_i]

            Rui = scale(batch_r, 0.0, 1.0, train_set.min_rating, train_set.max_rating)
            Rui = torch.tensor(Rui, device=device)

            Xui = torch.sigmoid(
                torch.sum(U_u * V_i, dim=1) + torch.sum(P_u * f_i.mm(E), dim=1)
            )

            loss = _l2_loss(Rui - Xui)
            reg = (
                lambda_u * _l2_loss(U_u)
                + lambda_v * _l2_loss(V_i)
                + lambda_p * _l2_loss(P_u)
                + lambda_e * _l2_loss(E)
            )
            loss = loss + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(batch_u)
            if count % (batch_size * 10) == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))
            progress_bar.update(1)
        progress_bar.close()

        if verbose:
            print(sum_loss)

    res = {
        "U": U.data.cpu().numpy(),
        "V": V.data.cpu().numpy(),
        "P": P.data.cpu().numpy(),
        "E": E.data.cpu().numpy(),
        "Q": F.mm(E).data.cpu().numpy(),
    }

    return res
