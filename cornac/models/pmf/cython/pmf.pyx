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

# cython: language_level=3

from libc.math cimport exp
from libc.math cimport sqrt

import numpy as np

from ...utils import get_rng
from ...utils.init_utils import normal

#Sigmoid function
cdef float sigmoid(float z):
    cdef float s
    cdef float MAX_EXP = 6.0
    if z > MAX_EXP:
        return 1.0
    else:
        if z < -MAX_EXP:
            return 0.0
        else:
            s = 1.0 / (1.0 + exp(-z))
            return s


def _init_factors(n, d, k, init_params, seed):
    rng = get_rng(seed)

    U = init_params.get('U', None)
    if U is None:
        U = normal((n, k), mean=0.0, std=0.001, random_state=rng, dtype=np.double)
    
    V = init_params.get('V', None)
    if V is None:
        V = normal((d, k), mean=0.0, std=0.001, random_state=rng, dtype=np.double)

    return U, V


#PMF (Gaussian linear-model version), SGD_RMSProp optimizer
def pmf_linear(int[:] uid, int[:] iid, float[:] rat, int n_users, int n_items, int n_ratings,
               int k, int n_epochs = 100, float lambda_reg = 0.001, float learning_rate = 0.001, float gamma = 0.9,
               init_params = {}, verbose = False, seed = None):
  
    #some useful variables
    cdef:
        double[:] loss = np.full(n_epochs, 0.0)
        int n = n_users
        int d = n_items
        int nnz = n_ratings
        double[:,:] U
        double[:,:] V
        double[:,:] cache_u = np.zeros((n,k))
        double[:,:] cache_v = np.zeros((d,k))
        double[:,:] grad_u = np.zeros((n,k))
        double[:,:] grad_v = np.zeros((d,k))
        double eps = 1e-8
        int u_, i_, k_, r, epoch
        double val, s, e, norm_u, norm_v
        
    U, V = _init_factors(n, d, k, init_params, seed)
  
    #Optimization
    for epoch in range(n_epochs):
        for r in range(nnz):
            u_, i_, val = uid[r], iid[r], rat[r]
            s = 0.0
            for k_ in range(k):
                s+= U[u_,k_]*V[i_,k_]
            e = val - s  # Error for the obseved rating u, i, val
            
            # update user factors
            for k_ in range(k):
                grad_u[u_,k_] = e * V[i_,k_]- lambda_reg * U[u_,k_]
                cache_u[u_,k_] = gamma * cache_u[u_,k_] + (1 - gamma) * (grad_u[u_,k_]*grad_u[u_,k_])
                U[u_,k_] += learning_rate * (grad_u[u_,k_]/(sqrt(cache_u[u_,k_])+eps)) # Update the user factor, better to reweight the L2 regularization terms acoording the number of ratings per-user
            
            # update item factors
            for k_ in range(k):
                grad_v[i_,k_] = e * U[u_,k_] - lambda_reg * V[i_, k_]
                cache_v[i_,k_] = gamma * cache_v[i_,k_] + (1 - gamma) * (grad_v[i_,k_]*grad_v[i_,k_])            
                V[i_,k_] += learning_rate * (grad_v[i_,k_]/(sqrt(cache_v[i_,k_]) + eps))            
            
            norm_u = 0.0
            norm_v = 0.0
            for k_ in range(k):
                norm_u += U[u_,k_]*U[u_,k_]
                norm_v += V[i_,k_]*V[i_,k_]
            
            loss[epoch]+= e*e  + lambda_reg * (norm_u + norm_v)

        if verbose:
            print('epoch %i, loss: %f' % (epoch, loss[epoch]))
 
    res = {'U':U,'V':V,'loss': loss}
    
    return res


#PMF (Gaussian non-linear model version using sigmoid function)  SGD_RMSProp optimizer
def pmf_non_linear(int[:] uid, int[:] iid, float[:] rat, int n_users, int n_items, int n_ratings,
                   int k, int n_epochs = 100, float lambda_reg = 0.001, float learning_rate = 0.001, float gamma = 0.9,
                   init_params = None, verbose = False, seed = None):
  
    #some useful variables
    cdef:
        double[:] loss = np.full(n_epochs, 0.0)
        int n = n_users
        int d = n_items
        double[:,:] U
        double[:,:] V
        double[:,:] cache_u = np.zeros((n,k))
        double[:,:] cache_v = np.zeros((d,k))
        double[:,:] grad_u = np.zeros((n,k))
        double[:,:] grad_v = np.zeros((d,k))
        double eps = 1e-8
        int u_, i_, r, k_, epoch 
        double val, s, e, we, sg, norm_u, norm_v
  
    U, V = _init_factors(n, d, k, init_params, seed)
  
    #Optimization
    for epoch in range(n_epochs):
        for r in range(n_ratings):
            u_, i_, val = uid[r], iid[r], rat[r]

            s = 0.0
            for k_ in range(k):
                s += U[u_, k_] * V[i_, k_]
            sg = sigmoid(s)
            e = (val - sg)  # Error for the obseved rating u_, i_, val
            we = e * sg * (1. - sg)  # Weighted error

            # update user factors
            for k_ in range(k):
                grad_u[u_, k_] = we * V[i_, k_] - lambda_reg * U[u_, k_]
                cache_u[u_, k_] = gamma * cache_u[u_, k_] + (1 - gamma) * (grad_u[u_, k_] * grad_u[u_, k_])
                U[u_, k_] += learning_rate * (grad_u[u_, k_] / (sqrt(cache_u[u_, k_]) + eps))  # Update the user factor, better to reweight the L2 regularization terms acoording the number of ratings per-user

            # update item factors
            for k_ in range(k):
                grad_v[i_, k_] = we * U[u_, k_] - lambda_reg * V[i_, k_]
                cache_v[i_, k_] = gamma * cache_v[i_, k_] + (1 - gamma) * (grad_v[i_, k_] * grad_v[i_, k_])
                V[i_, k_] += learning_rate * (grad_v[i_, k_] / (sqrt(cache_v[i_, k_]) + eps))

            norm_u = 0.0
            norm_v = 0.0
            for k_ in range(k):
                norm_u += U[u_, k_] * U[u_, k_]
                norm_v += V[i_, k_] * V[i_, k_]

            loss[epoch] += e * e + lambda_reg * (norm_u + norm_v)

        if verbose:
            print('epoch %i, loss: %f' % (epoch, loss[epoch]))

    res = {'U':U,'V':V,'loss': loss}
    
    return res
