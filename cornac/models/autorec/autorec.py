# -*- coding: utf-8 -*-
"""
@author: Guo Jingyao <jyguo@smu.edu.sg>
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pdb

def autorec(train_set, data, k, lamda=0.01, n_epochs=100, learning_rate=0.001, batch_size=50, g_act="Sigmoid", f_act="Identity", init_params=None):

    # user base autorec
    X = train_set.matrix

    d = X.shape[1]
    n = X.shape[0]
    rating = torch.sparse.FloatTensor(torch.LongTensor([X.tocoo().row, X.tocoo().col]), torch.FloatTensor(X.data), torch.Size([n, d]))
    M = torch.sparse.FloatTensor(torch.LongTensor([X.tocoo().row, X.tocoo().col]), torch.ones(len(X.data)), torch.Size([n, d]))

    # Initial parameters
    if init_params['V'] is None:
        V = torch.randn(k, d, requires_grad=True)
    else:
        V = init_params['V']
        newV = np.zeros((k, d))
        for oldidx, newidx in train_set._iid_map.items():
            newV[:, newidx]=V[:, int(oldidx)]
        # V = torch.from_numpy(newV).float()
        V = torch.tensor(newV, dtype=torch.float32, requires_grad=True)

    if init_params['W'] is None:
        W = torch.randn(d, k, requires_grad=True)
    else:
        W = init_params['W']
        newW = np.zeros((d, k))
        for oldidx, newidx in train_set._iid_map.items():
            newW[newidx, :]=W[int(oldidx), :]
        # W = torch.from_numpy(newW).float()
        W = torch.tensor(newW, dtype=torch.float32, requires_grad=True)

    if init_params['mu'] is None:
        mu = torch.ones(k, 1, requires_grad=False)
    else:
        mu = init_params['mu']
        mu = torch.from_numpy(mu)

    if init_params['b'] is None:
        b = torch.ones(d, 1, requires_grad=False)
    else:
        b = init_params['b']
        b = torch.from_numpy(b)

    if g_act == "Sigmoid":
        g_act = nn.Sigmoid()
    elif g_act == "Relu":
        g_act = nn.Relu()
    elif g_act == "Tanh":
        g_act = nn.Tanh()
    elif g_act == "Identity":
        g_act = nn.Dropout(0)
    elif g_act == "Elu":
        g_act = nn.Elu()
    else:
        raise NotImplementedError("Active function ERROR")

    if f_act == "Sigmoid":
        f_act = nn.Sigmoid()
    elif f_act == "Relu":
        f_act = nn.ReLU()
    elif f_act == "Tanh":
        f_act = nn.Tanh()
    elif f_act == "Identity":
        f_act = nn.Dropout(0)
    elif f_act == "Elu":
        f_act = nn.Elu()
    else:
        raise NotImplementedError("Active function ERROR")

    optimizer = torch.optim.Adam([V, W], lr=learning_rate)
    for epoch in range(n_epochs):

        # set batch size for each step, and shuffle the training data
        train_loader = torch.utils.data.DataLoader(np.arange(n), batch_size=batch_size, shuffle=True)
        for step, users in enumerate(train_loader):
            train, mask = prepareData(users, data, d)
            train = torch.from_numpy(train).float()
            mask = torch.from_numpy(mask).float()
            h = f_act(W.mm(g_act(V.mm(torch.t(train)) + mu.repeat(1, len(users)))) + b.repeat(1, len(users)))
            h = torch.t(h)
            loss = ((h - train) * mask).pow(2).sum() + lamda * (V.norm().pow(2) + W.norm().pow(2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        H = f_act(W.mm(g_act(V.mm(torch.t(rating).to_dense()) + mu.repeat(1, n))) + b.repeat(1, n))
        H = torch.t(H)
        L = ((H - rating.to_dense()) * (M.to_dense())).pow(2).sum() + lamda * (V.norm().pow(2) + W.norm().pow(2))
        print('epoch:', epoch, 'loss:', L)

    E = V.mm(torch.t(rating).to_dense()).data.numpy()
    W = W.data.numpy()
    V = V.data.numpy()
    mu = mu.data.numpy()
    b = b.data.numpy()

    res = {'W': W, 'V': V, 'mu': mu, 'b': b, 'E': E}
    return res

def prepareData(users, data, d):
    users = np.array(users)
    train = np.zeros((len(users), d))
    mask = np.zeros((len(users), d))

    for i, userid in enumerate(users):
        index = np.isin(data[:, 0], userid)
        items = data[index, 1].astype(int)
        rating = data[index, 2]
        train[i, items] = rating
        mask[i, items] = 1

    return train, mask

