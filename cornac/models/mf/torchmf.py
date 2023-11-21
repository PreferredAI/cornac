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

import torch
import torch.nn as nn
from tqdm.auto import trange


class MF_Pytorch(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        n_factors=10,
        global_mean=0,
        dropout=0,
        init_params={},
    ):
        super(MF_Pytorch, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.u_factors = nn.Embedding(n_users, n_factors)
        self.i_factors = nn.Embedding(n_items, n_factors)
        self.u_biases = nn.Embedding(n_users, 1)
        self.i_biases = nn.Embedding(n_items, 1)
        self.global_mean = global_mean
        self.dropout = nn.Dropout(p=dropout)

        self._init_params(init_params)

    def _init_params(self, init_params={}):
        if not init_params:
            nn.init.normal_(self.u_factors.weight, std=0.01)
            nn.init.normal_(self.i_factors.weight, std=0.01)
            self.u_biases.weight.data.fill_(0.0)
            self.i_biases.weight.data.fill_(0.0)
            return

        if "U" in init_params:
            self.u_factors.weight.data = torch.from_numpy(init_params["U"])
        if "V" in init_params:
            self.i_factors.weight.data = torch.from_numpy(init_params["V"])
        if "Bu" in init_params:
            self.u_biases.weight.data = torch.from_numpy(init_params["Bu"])
        if "Bi" in init_params:
            self.i_biases.weight.data = torch.from_numpy(init_params["Bi"])

    def forward(self, uids, iids):
        ues = self.u_factors(uids)
        uis = self.i_factors(iids)

        preds = self.u_biases(uids) + self.i_biases(iids) + self.global_mean
        preds += (self.dropout(ues) * self.dropout(uis)).sum(dim=1, keepdim=True)
        return preds.squeeze()

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users, items):
        with torch.no_grad():
            return self.forward(users, items)


loss_fn_dict = {
    "mse": nn.MSELoss(reduction="sum"),
}

optimizer_dict = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}


def learn(
    model,
    train_set,
    n_epochs,
    batch_size=256,
    learn_rate=0.01,
    reg=0,
    verbose=True,
    criteria="mse",
    optimizer="sgd",
    device=torch.device("cpu"),
):
    model = model.to(device)

    criteria = loss_fn_dict[criteria]
    optimizer = optimizer_dict[optimizer](params=model.parameters(), lr=learn_rate, weight_decay=reg)

    progress_bar = trange(1, n_epochs + 1, disable=not verbose)
    for _ in progress_bar:
        sum_loss = 0.0
        count = 0
        for batch_id, (u_batch, i_batch, r_batch) in enumerate(train_set.uir_iter(batch_size, shuffle=True)):
            u_batch = torch.from_numpy(u_batch).to(device)
            i_batch = torch.from_numpy(i_batch).to(device)
            r_batch = torch.tensor(r_batch, dtype=torch.float).to(device)

            preds = model(u_batch, i_batch)
            loss = criteria(preds, r_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.data.item()
            count += len(u_batch)

            if batch_id % 10 == 0:
                progress_bar.set_postfix(loss=(sum_loss / count))

    return model
