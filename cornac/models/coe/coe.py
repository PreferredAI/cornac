# -*- coding: utf-8 -*-
"""
@author: Dung D. Le (Andrew) <ddle.2015@smu.edu.sg> 
"""

import numpy as np
import scipy.sparse as sp
import random
import torch
from ...utils.data_utils import Dataset

"""Firstly, we define a helper function to generate\sample training ordinal triplets:
   Step 1:  
   given rated item i, randomly choose item j and check whether rating of j is lower than i, 
   if not randomly sample another item. 
   each row of the sampled data in the following form:
        [userId itemId_i itemId_j]
   for each user u, he/she prefers item i over item j.
   """


def sample_triplet(X, batch_size):
    sampled_data = np.zeros((batch_size, 3), dtype=np.int)

    count = 0
    while count < batch_size:
        u = random.randint(0, X.shape[0] - 1)
        u_row = X.getrow(u)
        _, u_nz = u_row.nonzero()
        min_rating = u_row[:, u_nz].todense().min()

        i = u_nz[random.randint(0, len(u_nz) - 1)]
        ratingi = u_row[:, i]

        if ratingi > min_rating:
            j = u_nz[random.randint(0, len(u_nz) - 1)]

            while u_row[:, j] >= ratingi:
                j = u_nz[random.randint(0, len(u_nz) - 1)]

            sampled_data[count, :] = [u, i, j]
            count += 1

    print("Done sampling")
    return sampled_data


def coe(X, k, lamda=0.05, n_epochs=150, learning_rate=0.001, batch_size=1000, init_params=None):
    # Data = Dataset(data)

    # Initial user factors
    if init_params['U'] is None:
        U = torch.randn(X.shape[0], k, requires_grad=True, device="cuda")
    else:
        U = init_params['U']
        U = torch.from_numpy(U)

    # Initial item factors
    if init_params['V'] is None:
        V = torch.randn(X.shape[1], k, requires_grad=True, device="cuda")
    else:
        V = init_params['V']
        V = torch.from_numpy(V)

    optimizer = torch.optim.Adam([U, V], lr=learning_rate)
    for epoch in range(n_epochs):
        #        num_steps = int(Data.data.shape[0]/batch_size)

        #       for i in range(1, num_steps + 1):
        #          batch_c,_ = Data.next_batch(batch_size)
        sampled_batch = sample_triplet(X, batch_size)

        regU = U[sampled_batch[:, 0], :]
        regI = V[sampled_batch[:, 1], :]
        regJ = V[sampled_batch[:, 2], :]

        regU_unq = U[np.unique(sampled_batch[:, 0]), :]
        regI_unq = V[np.unique(sampled_batch[:, 1:]), :]

        Scorei = torch.norm(regU - regI, dim=1)
        Scorej = torch.norm(regU - regJ, dim=1)

        loss = lamda * (regU_unq.norm().pow(2) + regI_unq.norm().pow(2)) - torch.log(
            torch.sigmoid(Scorej - Scorei)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch:', epoch, 'loss:', loss)

    U = U.data.cpu().numpy()
    V = V.data.cpu().numpy()

    res = {'U': U, 'V': V}

    return res
