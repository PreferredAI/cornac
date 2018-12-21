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


def vbpr(X, data, k, d, aux_info, lamda=0.01, n_epochs=100, learning_rate=0.001, batch_size=10000, init_params=None):
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

    if aux_info.all() != None:
        aux_info = torch.from_numpy(aux_info).float()
        print("add img feature info")
        if init_params['E'] is None:
            E = torch.randn(d, aux_info.shape[1], requires_grad=True)
        else:
            E = init_params['E']
            E = torch.tensor(E, requires_grad=True)

        if init_params['Ue'] is None:
            Ue = torch.randn(X.shape[0], d, requires_grad=True)
        else:
            Ue = init_params['Ue']
            Ue = torch.tensor(Ue, requires_grad=True)

    optimizer = torch.optim.Adam([U, V, E, Ue], lr=learning_rate)

    for epoch in range(n_epochs):

        num_steps = int(Data.data.shape[0] / batch_size)
        for i in range(1, num_steps + 1):
            batch_c, _ = Data.next_batch(batch_size)
            sampled_batch = sample_data(X, batch_c)
            regU = U[sampled_batch[:, 0], :]
            regVi = V[sampled_batch[:, 1], :]
            regVj = V[sampled_batch[:, 2], :]
            Ri = torch.sum(regU * regVi, dim=1) + torch.sum(
                Ue[sampled_batch[:, 0], :] * (aux_info[sampled_batch[:, 1], :].mm(E.t())), dim=1)
            Rj = torch.sum(regU * regVj, dim=1) + torch.sum(
                Ue[sampled_batch[:, 0], :] * (aux_info[sampled_batch[:, 2], :].mm(E.t())), dim=1)
            loss = (lamda * (regU.norm().pow(2) + regVi.norm().pow(2) + regVj.norm().pow(2))
                    - torch.log(torch.sigmoid(Ri - Rj) + 1e-10).sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch:', epoch, 'loss:', loss)

    U = U.data.numpy()
    V = V.data.numpy()
    E = E.data.numpy()
    Ue = Ue.data.numpy()

    res = {'U': U, 'V': V, 'E': E, 'Ue': Ue}

    return res
