# -*- coding: utf-8 -*-
"""
@author: Aghiles Salah
"""
import numpy as np
from libc.math cimport exp


#Sigmoid function
cdef double sigmoid(double z):
    cdef double s
    cdef double MAX_EXP = 6.0
    if z > MAX_EXP:
        return 1.0
    else:
        if z < -MAX_EXP:
            return 0.0
        else:
            s = 1.0 / (1.0 + exp(-z))
            return s


#PMF (Gaussian linear-model version), SGD_RMSProp optimizer
def pmf_linear(X,n_X,d_X,k, n_epochs = 100, lamda = 0.001,learning_rate=0.001,gamma = 0.9, init_params = None):

    #some useful variables
    loss = np.full(n_epochs, 0.0)
    n = n_X
    d = d_X
    cache_u = np.zeros((n,k))
    cache_v = np.zeros((d,k))
    grad_u = np.zeros((n,k))
    grad_v = np.zeros((d,k))
    eps = 1e-8
    #Parameter initialization
    #User factors
    if init_params['U'] is None:
        U = np.random.normal(loc=0.0, scale=0.001, size=n*k).reshape(n,k)
    else:
        U = init_params['U']
    
    #Item factors
    if init_params['V'] is None:
        V = np.random.normal(loc=0.0, scale=0.001, size=d*k).reshape(d,k)
    else:
        V = init_params['V']
  
    #Optimization
    
    for epoch in range(n_epochs):
        for u_, i_, val in X:
            u_, i_ = int(u_), int(i_)
            e = val - np.dot(U[u_,:], V[i_,:].T)  # Error for the obseved rating u, i, val
            #update user factors
            grad_u[u_,:] = e * V[i_,:]- lamda * U[u_, :]
            cache_u[u_,:] = gamma * cache_u[u_,:] + (1 - gamma) * (grad_u[u_,:]*grad_u[u_,:])
            U[u_,:] += learning_rate * (grad_u[u_,:]/(np.sqrt(cache_u[u_,:])+eps) ) # Update the user factor,
            #update item factors
            grad_v[i_,:] = e * U[u_,:] - lamda * V[i_, :]
            cache_v[i_,:] = gamma * cache_v[i_,:] + (1 - gamma) * (grad_v[i_,:]*grad_v[i_,:])            
            V[i_,:] += learning_rate * (grad_v[i_,:]/(np.sqrt(cache_v[i_,:])+eps) )  # Update item factor 
            loss[epoch]+= e*e  + lamda * (np.dot(U[u_, :].T, U[u_, :]) + np.dot(V[i_, :].T, V[i_, :]))
        print("loss:",loss[epoch])    
 
    res = {'U':U,'V':V,'loss': loss}
    
    return res



"""
#PMF (Gaussian linear-model version) SGD optimizer
def pmf(X,n_X,d_X,k, n_epochs = 100, lamda = 0.01,learning_rate=0.001, init_params = None):
  
    print("I am pmf")
    ### data preparation
    loss = np.full(n_epochs, 0.0)
    n = n_X
    d = d_X

    cache_u = np.zeros((n,k))
    cache_v = np.zeros((d,k))
    grad_u = np.zeros((n,k))
    grad_v = np.zeros((d,k))
    double gamma = 0.9
    double eps = 1e-8
    #Parameter initialization
    #User factors
    if init_params['U'] is None:
        U = np.random.normal(loc=0.0, scale=0.1, size=n*k).reshape(n,k)
    else:
        U = init_params['U']
    
    #Item factors
    if init_params['V'] is None:
        V = np.random.normal(loc=0.0, scale=0.1, size=d*k).reshape(d,k)
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
"""



#PMF (Gaussian non-linear model version using sigmoid function)  SGD_RMSProp optimizer
def pmf_non_linear(X,n_X,d_X,k, n_epochs = 100, lamda = 0.001,learning_rate=0.001,gamma = 0.9, init_params = None):
  
    #some useful variables
    loss = np.full(n_epochs, 0.0)
    n = n_X
    d = d_X
    cache_u = np.zeros((n,k))
    cache_v = np.zeros((d,k))
    grad_u = np.zeros((n,k))
    grad_v = np.zeros((d,k))
    eps = 1e-8
  
    #Parameter initialization
    #User factors
    if init_params['U'] is None:
        U = np.random.normal(loc=0.0, scale=0.001, size=n*k).reshape(n,k)
    else:
        U = init_params['U']
    
    #Item factors
    if init_params['V'] is None:
        V = np.random.normal(loc=0.0, scale=0.001, size=d*k).reshape(d,k)
    else:
        V = init_params['V']
  
    #Optimization
    
    for epoch in range(n_epochs):
        for u_, i_, val in X:
            u_, i_ = int(u_), int(i_)
            sg = sigmoid(np.dot(U[u_,:], V[i_,:].T))
            e = (val - sg)      #Error for the obseved rating u, i, val
            we= e*sg*(1.-sg)   #Weighted error for the obseved rating u, i, val
            grad_u[u_,:] = we * V[i_,:]- lamda * U[u_, :]
            cache_u[u_,:] = gamma * cache_u[u_,:] + (1 - gamma) * (grad_u[u_,:]*grad_u[u_,:])
            U[u_,:] += learning_rate * (grad_u[u_,:]/(np.sqrt(cache_u[u_,:])+eps) ) # Update the user factor, better to reweight the L2 regularization terms acoording the number of ratings per-user 
            #update item factors
            grad_v[i_,:] = we * U[u_,:] - lamda * V[i_, :]
            cache_v[i_,:] = gamma * cache_v[i_,:] + (1 - gamma) * (grad_v[i_,:]*grad_v[i_,:])            
            V[i_,:] += learning_rate * (grad_v[i_,:]/(np.sqrt(cache_v[i_,:])+eps) )  # Update item factor             
            loss[epoch]+= e*e  + lamda * (np.dot(U[u_, :].T, U[u_, :]) + np.dot(V[i_, :].T, V[i_, :]))
        print('epoch %i, loss: %f' % (epoch, loss[epoch]))
    res = {'U':U,'V':V,'loss': loss}
    
    return res



