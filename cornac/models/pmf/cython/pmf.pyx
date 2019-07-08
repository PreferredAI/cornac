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


#PMF (Gaussian linear-model version), SGD_RMSProp optimizer
def pmf_linear(int[:] uid, int[:] iid, float[:] rat, int n_users, int n_items, int n_ratings,
               int k, int n_epochs = 100, float lamda = 0.001, float learning_rate = 0.001, float gamma = 0.9,
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
        int u_, i_, j, epoch
        double val, s, e, norm_u, norm_v
        
    # Initialize user factors
    rng = get_rng(seed)
    U = init_params.get('U', normal((n,k), mean=0.0, std=0.001, random_state=rng, dtype=np.double))
    V = init_params.get('V', normal((d,k), mean=0.0, std=0.001, random_state=rng, dtype=np.double))
  
    #Optimization
    for epoch in range(n_epochs):
        for r in range(nnz):
            u_, i_, val = uid[r], iid[r], rat[r]
            s = 0.0
            for j in range(k):
                s+= U[u_,j]*V[i_,j]
            e = val - s  # Error for the obseved rating u, i, val
            
            # update user factors
            for j in range(k):
                grad_u[u_,j] = e * V[i_,j]- lamda * U[u_,j]
                cache_u[u_,j] = gamma * cache_u[u_,j] + (1 - gamma) * (grad_u[u_,j]*grad_u[u_,j])
                U[u_,j] += learning_rate * (grad_u[u_,j]/(sqrt(cache_u[u_,j])+eps)) # Update the user factor, better to reweight the L2 regularization terms acoording the number of ratings per-user
            
            # update item factors
            for j in range(k):
                grad_v[i_,j] = e * U[u_,j] - lamda * V[i_, j]
                cache_v[i_,j] = gamma * cache_v[i_,j] + (1 - gamma) * (grad_v[i_,j]*grad_v[i_,j])            
                V[i_,j] += learning_rate * (grad_v[i_,j]/(sqrt(cache_v[i_,j]) + eps))            
            
            norm_u = 0.0
            norm_v = 0.0
            for j in range(k):
                norm_u += U[u_,j]*U[u_,j]
                norm_v += V[i_,j]*V[i_,j]
            
            loss[epoch]+= e*e  + lamda * (norm_u + norm_v)

        if verbose:
            print('epoch %i, loss: %f' % (epoch, loss[epoch]))
 
    res = {'U':U,'V':V,'loss': loss}
    
    return res


#PMF (Gaussian non-linear model version using sigmoid function)  SGD_RMSProp optimizer
def pmf_non_linear(int[:] uid, int[:] iid, float[:] rat, int n_users, int n_items, int n_ratings,
                   int k, int n_epochs = 100, float lamda = 0.001, float learning_rate = 0.001, float gamma = 0.9,
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
        int u_, i_, j, epoch
        double val, s, e, norm_u, norm_v
  
    rng = get_rng(seed)
    U = init_params.get('U', normal((n,k), mean=0.0, std=0.001, random_state=rng, dtype=np.double))
    V = init_params.get('V', normal((d,k), mean=0.0, std=0.001, random_state=rng, dtype=np.double))
  
    #Optimization
    for epoch in range(n_epochs):
        for r in range(nnz):
            u_, i_, val = uid[r], iid[r], rat[r]
            
            s = 0.0
            for j in range(k):
                s+= U[u_,j]*V[i_,j]
            sg = sigmoid(s)
            e = (val - sg)     #Error for the obseved rating u, i, val
            we= e*sg*(1.-sg)   #Weighted error for the obseved rating u, i, val
            
            # update user factors
            for j in range(k):
                grad_u[u_,j] = we * V[i_,j]- lamda * U[u_,j]
                cache_u[u_,j] = gamma * cache_u[u_,j] + (1 - gamma) * (grad_u[u_,j]*grad_u[u_,j])
                U[u_,j] += learning_rate * (grad_u[u_,j]/(sqrt(cache_u[u_,j])+eps)) # Update the user factor, better to reweight the L2 regularization terms acoording the number of ratings per-user 
            
            # update item factors
            for j in range(k):
                grad_v[i_,j] = we * U[u_,j] - lamda * V[i_, j]
                cache_v[i_,j] = gamma * cache_v[i_,j] + (1 - gamma) * (grad_v[i_,j]*grad_v[i_,j])            
                V[i_,j] += learning_rate * (grad_v[i_,j]/(sqrt(cache_v[i_,j]) + eps))  
                
            norm_u = 0.0
            norm_v = 0.0
            for j in range(k):
                norm_u += U[u_,j]*U[u_,j]
                norm_v += V[i_,j]*V[i_,j]
            
            loss[epoch]+= e*e  + lamda * (norm_u + norm_v)

        if verbose:
            print('epoch %i, loss: %f' % (epoch, loss[epoch]))

    res = {'U':U,'V':V,'loss': loss}
    
    return res
