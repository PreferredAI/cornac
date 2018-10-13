# -*- coding: utf-8 -*-
"""
@author: Dung D. Le (Andrew) <ddle.2015@smu.edu.sg> 
"""

import numpy as np
import random
import torch
from torch.utils.data import DataLoader

"""Firstly, we define a helper function to generate\sample training ordinal triplets:
   Step 1:  
   given rated item i, randomly choose item j and check whether rating of j is missing or lower than i, 
   if not randomly sample another item. 
   each row of the sampled data in the following form:
        [userId itemId_i itemId_j rating_i rating_j]
   for each user u, he/she prefers item i over item j.
   """
def sampleData(X, data):
    X = X.todense()
    sampled_data = np.zeros((data.shape[0], 5), dtype=np.int)
    data = data.astype(int)

    for k in range(0, data.shape[0]):
        u = data[k, 0]
        i = data[k, 1]
        ratingi = data[k, 2]
        j = random.randint(0, X.shape[0])

        while X[u, j] > ratingi:
            j = random.randint(0, data.shape[1])

        sampled_data[k, :] = [u, i, j, ratingi, X[u, j]]

    return sampled_data


def ibpr(X, data, k, lamda = 0.005, n_epochs=150, learning_rate=0.001,batch_size = 100, init_params=None):

    #Initial user factors
    if init_params['U'] is None:
        U = torch.randn(X.shape[0], k, requires_grad=True)
    else:
        U = init_params['U']

    #Initial item factors
    if init_params['V'] is None:
        V = torch.randn(X.shape[1], k, requires_grad=True)
    else:
        V = init_params['V']
    
    optimizer = torch.optim.Adam([U, V], lr=learning_rate)
    for epoch in range(n_epochs):
        # for each epoch, randomly sample training ordinal triplets
        Data = sampleData(X, data)
        # set batch size for each step, and shuffle the training data
        train_loader = torch.utils.data.DataLoader(Data, batch_size=batch_size, shuffle=True)
        for step, batch_data in enumerate(train_loader):
            
            U_norm     = U / U.norm(dim = 1)[:, None]
            V_norm     = V / V.norm(dim = 1)[:, None] 
            angularSim = torch.acos(torch.clamp(U_norm.mm(V_norm.t()), -1 + 1e-7, 1 - 1e-7))  
            batch_data = np.array(batch_data)
            regU = U[batch_data[:, 0], :]
            regV = V[np.unique(np.append(batch_data[:, 1], batch_data[:, 2])), :]

            Scorei = angularSim[batch_data[:, 0], batch_data[:, 1]]
            Scorej = angularSim[batch_data[:, 0], batch_data[:, 2]]

            loss = lamda * (torch.trace(regU.mm(regU.t())) + torch.trace(regV.mm(regV.t()))) - torch.log(
                torch.sigmoid(Scorej.add(Scorei * -1))).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch:',epoch,'loss:', loss)
    
    # since the user's preference is defined by the angular distance, we can normalize the user/item vectors without changing the ranking
    U = torch.nn.functional.normalize(U, p = 2, dim=1)
    V = torch.nn.functional.normalize(V, p = 2, dim=1)
    U = U.data.numpy()
    V = V.data.numpy()

    res = {'U': U, 'V': V}

    return res

