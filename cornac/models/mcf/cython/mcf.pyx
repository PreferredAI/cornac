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

from ...utils.init_utils import normal
from ...utils import get_rng


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



# MCF SGD_RMSProp optimizer
def mcf(int[:] rat_uid, int[:] rat_iid, float[:] rat_val, int[:] net_iid, int[:] net_jid, float[:] net_val,
        int n_users, int n_items, int n_ratings, int n_edges, int k, int n_epochs = 100, float lamda = 0.001,
        float learning_rate = 0.001, float gamma = 0.9, init_params = {}, verbose = False, seed = None):
  
    #some useful variables
    cdef:
        double[:] loss = np.full(n_epochs, 0.0)
        int n = n_users
        int d = n_items
        double[:,:] U
        double[:,:] V
        double[:,:] Z
        double[:,:] cache_u = np.zeros((n,k))
        double[:,:] cache_v = np.zeros((d,k))
        double[:,:] cache_z = np.zeros((d,k))
        double[:,:] grad_u = np.zeros((n,k))
        double[:,:] grad_v = np.zeros((d,k))
        double[:,:] grad_z = np.zeros((d,k))
        double eps = 1e-8
        int u_, i_, j_, k_, r, ed, epoch
        double val, s, e, norm_u, norm_v
  
    # Initialize factors
    rng = get_rng(seed)
   
    U = init_params.get('U', None)
    if U is None:
        U = normal((n, k), mean=0.0, std=0.001, random_state=rng, dtype=np.double)
   
    V = init_params.get('V', None)
    if V is None:
        V = normal((d, k), mean=0.0, std=0.001, random_state=rng, dtype=np.double)
    
    Z = init_params.get('Z', None)
    if Z is None:
        Z = normal((d, k), mean=0.0, std=0.001, random_state=rng, dtype=np.double)
  
    #Optimization
    for epoch in range(n_epochs):   
        
        for ed in range(n_edges):
            i_, j_, val = net_iid[ed], net_jid[ed], net_val[ed]
            
            s = 0.0
            for k_ in range(k):
                s+= V[i_,k_]*Z[j_,k_]
            sg = sigmoid(s)
            err = (val - sg)     #Error for the obseved rating u_, i_, val
            werr = err*sg*(1.-sg)   #Weighted error
            
            # update item factors
            for k_ in range(k):
                grad_v[i_,k_] = werr * Z[j_,k_] - lamda * V[i_,k_]
                cache_v[i_,k_] = gamma * cache_v[i_,k_] + (1 - gamma) * (grad_v[i_,k_]*grad_v[i_,k_])            
                V[i_,k_] += learning_rate * (grad_v[i_,k_]/(sqrt(cache_v[i_,k_]) + eps))
                
            # update Also-Viewed item factors
            for k_ in range(k):
                grad_z[j_,k_] = werr * V[i_,k_] - lamda * Z[j_,k_]
                cache_z[j_,k_] = gamma * cache_z[j_,k_] + (1 - gamma) * (grad_z[j_,k_]*grad_z[j_,k_])            
                Z[j_,k_] += learning_rate * (grad_z[j_,k_]/(sqrt(cache_z[j_,k_]) + eps))  
                
            norm_z = 0.0
            norm_v = 0.0
            for k_ in range(k):
                norm_z += Z[j_,k_]*Z[j_,k_]
                norm_v += V[i_,k_]*V[i_,k_]
            
            loss[epoch]+= err*err  + lamda * (norm_z + norm_v)
        
        for r in range(n_ratings):
            u_, i_, val = rat_uid[r], rat_iid[r], rat_val[r]
            
            s = 0.0
            for k_ in range(k):
                s+= U[u_,k_]*V[i_,k_]
            sg = sigmoid(s)
            err = (val - sg)     #Error for the obseved rating u_, i_, val
            werr = err*sg*(1.-sg)   #Weighted error
            
            # update user factors
            for k_ in range(k):
                grad_u[u_,k_] = werr * V[i_,k_]- lamda * U[u_,k_]
                cache_u[u_,k_] = gamma * cache_u[u_,k_] + (1 - gamma) * (grad_u[u_,k_]*grad_u[u_,k_])
                U[u_,k_] += learning_rate * (grad_u[u_,k_]/(sqrt(cache_u[u_,k_])+eps)) # Update the user factor, better to reweight the L2 regularization terms acoording the number of ratings per-user 
            
            # update item factors
            for k_ in range(k):
                grad_v[i_,k_] = werr * U[u_,k_] - lamda * V[i_,k_]
                cache_v[i_,k_] = gamma * cache_v[i_,k_] + (1 - gamma) * (grad_v[i_,k_]*grad_v[i_,k_])            
                V[i_,k_] += learning_rate * (grad_v[i_,k_]/(sqrt(cache_v[i_,k_]) + eps))  
                
            norm_u = 0.0
            norm_v = 0.0
            for k_ in range(k):
                norm_u += U[u_,k_]*U[u_,k_]
                norm_v += V[i_,k_]*V[i_,k_]
            
            loss[epoch]+= err*err  + lamda * (norm_u + norm_v)

        if verbose:
            print('epoch %i, loss: %f' % (epoch, loss[epoch]))

    res = {'U':U,'V':V, 'Z': Z,'loss': loss}
    
    return res
