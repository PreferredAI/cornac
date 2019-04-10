# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

from tqdm import tqdm
from ...utils.common import scale
import numpy as np


try:
    import torch
except ImportError:
    torch = None



def _load_or_randn(size, init_values, device):
    if init_values is None:
        tensor = np.random.normal(loc=0.0, scale=0.001, size=size)
        tensor = torch.tensor(tensor, device=device, requires_grad=True, dtype=torch.double)
    else:
        tensor = torch.tensor(init_values, requires_grad=True, device=device)
    return tensor


def _l2_loss(*tensors):
    l2_loss = 0
    for tensor in tensors:
        l2_loss += torch.sum(tensor ** 2) / 2
    return l2_loss

def vmf(train_set, item_feature, k, d, n_epochs, batch_size, lambda_u, lambda_v,
          lambda_p, lambda_e, learning_rate, gamma, init_params, use_gpu, verbose):

    from ...utils.init_utils import xavier_uniform
    device = None
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
      
    F = torch.from_numpy(item_feature).double().to(device)
    
    f_dim = train_set.item_image.feature_dim
    n_users = train_set.num_users
    n_items = train_set.num_items
    
    # preparing parameters
    U = _load_or_randn((n_users, k), init_values= init_params['U'], device = device)
    V = _load_or_randn((n_items, k), init_values= init_params['V'], device = device)
    P = _load_or_randn((n_users, d), init_values= init_params['P'], device = device)
    E = _load_or_randn((f_dim,d), init_values= init_params['E'], device = device)

    # optimizer
    optimizer = torch.optim.RMSprop([U, V, P, E], lr= learning_rate, alpha=0.9)
        
    print('this is the new one')
    for epoch in range(1, n_epochs + 1):
        sum_loss = 0.
        count = 0
        progress_bar = tqdm(total=train_set.num_batches(batch_size),
                            desc='Epoch {}/{}'.format(epoch, n_epochs),
                            disable=not verbose)
        
        for batch_u, batch_i, batch_r in train_set.uir_iter(batch_size, shuffle=True):
            U_u = U[batch_u]
            P_u = P[batch_u]
            V_i = V[batch_i]
            f_i = F[batch_i]
            
            Rui = scale(batch_r, 0., 1., train_set.min_rating, train_set.max_rating)
            Rui = torch.tensor(Rui, dtype=torch.double)
            
            Xui = torch.sigmoid(torch.sum(U_u * V_i, dim=1) + torch.sum(P_u * f_i.mm(E), dim = 1)) 
            
            loss = _l2_loss(Rui - Xui)
            reg = lambda_u * _l2_loss(U_u) + lambda_v * _l2_loss(V_i) \
                  + lambda_p * _l2_loss(P_u) + lambda_e * _l2_loss(E)
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
        print(sum_loss)
    res = {'U':U.data.cpu().numpy(),'V':V.data.cpu().numpy(), 'P': P.data.cpu().numpy(), 'E': E.data.cpu().numpy(), 'Q': F.mm(E).data.cpu().numpy()}

    return res
