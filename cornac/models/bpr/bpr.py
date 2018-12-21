# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao <jyguo@smu.edu.sg>
"""

import numpy as np
import random
import torch
from ...utils.data_utils import Dataset

"""Generate training data pairs:
   given rated item i, randomly choose item j and check whether rating of j is missing or lower than i, 
   if not randomly sample another item. 
   each row of the sampled data in the following form:
   [userId itemId_i itemId_j rating_i rating_j]
   for each user u, he/she prefers item i over item j.
   """


def sample_data(X, data):
    sampled_data = np.zeros((data.shape[0], 5), dtype=np.int)
    data = data.astype(int)

    for k in range(0, data.shape[0]):
        u = data[k, 0]
        i = data[k, 1]
        ratingi = data[k, 2]
        j = random.randint(0, X.shape[1] - 1)

        while X[u, j] > ratingi:
            j = random.randint(0, X.shape[1] - 1)

        sampled_data[k, :] = [u, i, j, ratingi, X[u, j]]

    return sampled_data


def bpr(X, data, k, lamda=0.01, n_epochs=100, learning_rate=0.001, batch_size=10000, init_params=None):
    Data = Dataset(data)

    # Initial user factors
    if init_params['U'] is None:
        U = torch.randn(X.shape[0], k, requires_grad=True)
    else:
        U = init_params['U']
        U = torch.tensor(U, requires_grad=True)

    # Initial item factors
    if init_params['V'] is None:
        V = torch.randn(X.shape[1], k, requires_grad=True)
    else:
        V = init_params['V']
        V = torch.tensor(V, requires_grad=True)

    optimizer = torch.optim.Adam([U, V], lr=learning_rate)

    for epoch in range(n_epochs):

        num_steps = int(Data.data.shape[0] / batch_size)
        for i in range(1, num_steps + 1):
            batch_c, _ = Data.next_batch(batch_size)
            # print(batch_c, idx)
            sampled_batch = sample_data(X, batch_c)
            regU = U[sampled_batch[:, 0], :]
            regVi = V[sampled_batch[:, 1], :]
            regVj = V[sampled_batch[:, 2], :]
            Ri = torch.diag(regU.mm(regVi.t()), 0)
            Rj = torch.diag(regU.mm(regVj.t()), 0)
            # print(torch.log(torch.sigmoid(regU.mm(regVi.t()) - regU.mm(regVj.t()))).sum())
            loss = (lamda * (regU.norm().pow(2) + regVi.norm().pow(2) + regVj.norm().pow(2))
                    - torch.log(torch.sigmoid(Ri - Rj)).sum())
            # loss = - torch.log(torch.sigmoid(Ri - Rj)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch:', epoch, 'loss:', loss)

    U = U.data.numpy()
    V = V.data.numpy()

    res = {'U': U, 'V': V}

    return res
