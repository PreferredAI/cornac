# -*- coding: utf-8 -*-
"""
@author: Aghiles Salah
"""
import numpy as np
#PMF (Gaussian linear-model version) implementation with SGD 
def pmf(X,n_X,d_X,k, n_epochs = 100, lamda = 0.01,learning_rate=0.001, init_params = None):
  
    print("I am pmf")
    ### data preparation
    loss = np.full(n_epochs, 0.0)
    n = n_X
    d = d_X

  
    #Parameter initialization
    #User factors
    if init_params['U'] is None:
        U = np.random.normal(loc=0.0, scale=1.0, size=n*k).reshape(n,k)
    else:
        U = init_params['U']
    
    #Item factors
    if init_params['V'] is None:
        V = np.random.normal(loc=0.0, scale=1.0, size=d*k).reshape(d,k)
    else:
        V = init_params['V']
  
    #Optimization
    
    for epoch in range(n_epochs):
        for u_, i_, val in X:
            u_, i_ = int(u_), int(i_)
            e = val - np.dot(U[u_,:], V[i_,:].T)  # Error for the obseved rating u, i, val
            U[u_,:] += learning_rate * ( e * V[i_,:]- lamda * U[u_, :]) # Update the user factor, this update  in not exactly faire need to reweight the L2 regularization terms acoording the number of ratings per-user 
            V[i_,:] += learning_rate * ( e * U[u_,:]- lamda * V[i_, :])  # Update item factor 
            loss[epoch]+= e*e  + lamda * (np.dot(U[u_, :].T, U[u_, :]) + np.dot(V[i_, :].T, V[i_, :]))
        print("loss:",loss[epoch])    
 
    res = {'U':U,'V':V,'loss': loss}
    
    return res