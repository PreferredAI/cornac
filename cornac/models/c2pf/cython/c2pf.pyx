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

from libcpp.vector cimport vector

import numpy as np
import scipy.sparse as sp

ctypedef vector[vector[double]] Mat
ctypedef vector[double] dVec
ctypedef vector[int] iVec

cdef extern from "cpp_c2pf.h":
    void c2pf_cpp(Mat &X, Mat &C, int &g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, Mat &L2_s, Mat &L2_r, Mat &L3_s, Mat &L3_r, dVec &T3_r, dVec &c_sum_c, dVec &util_sum, int maxiter, double at, double bt)
    void c2pf_cpp2(Mat &X, Mat &C, int &g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, Mat &L2_s, Mat &L2_r, Mat &L3_s, Mat &L3_r, dVec &T3_r, dVec &c_sum_c, dVec &util_sum, int maxiter,double at,double bt)
    void tc2pf_cpp(Mat &X, Mat &C, int &g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, Mat &L3_s, Mat &L3_r, dVec &T2_r, dVec &c_sum_c, dVec &util_sum, int maxiter, double at, double bt)
    void rc2pf_cpp(Mat &X, Mat &C, int &g, Mat &G_s, Mat &G_r, Mat &L2_s, Mat &L2_r, Mat L3_s, Mat L3_r, dVec &T3_r, dVec &c_sum_c, dVec &util_sum, int maxiter, double at, double bt)
    
#C2PF
cpdef c2pf(Mat &X, int n_X, int d_X, C,int n_C, int d_C, int &k, int &iter_max, init_param = None):
    
    n   = n_X
    d   = d_X
    d2  = d_C

    #Hyper parameter setting
    a  = 0.3
    a_ = 0.3
    a1 = 6.
    a1_t = 3.
    c  = 0.3
    c_ = 0.3
    c1 = 5.
    c1_t = 2.
    e  = 0.3
    a_t = 100.
    b_t = 100./0.5
    b_ = 1
    d_ = 1
    
    #Declare C++ variables
    cdef:
        Mat G_s, G_r, L_s, L_r, L2_s, L2_r, L3_s, L3_r
    
    
    
    #Parameters Initialization
    if init_param['G_s'] is None:
        G_s = np.random.gamma(100,scale=0.3/100,size=n*k).reshape(n,k)
    else:
        G_s = init_param['G_s']

    #G_s = sp.csc_matrix(G_s,dtype=np.float64)
  
    # rate gamma_uk matrix
    if init_param['G_r'] is None:
        G_r = np.random.gamma(100,scale=0.3/100, size=n*k).reshape(n,k)
    else:
        G_r = init_param['G_r']
    #G_r = sp.csc_matrix(G_r,dtype=np.float64)
  
    ## shape lamda_ik matrix (dgCMatrix)
    if init_param['L_s'] is None:
        L_s = np.random.gamma(100,scale=0.3/100, size=d*k).reshape(d,k)
                
    else:
        L_s = init_param['L_s']
    #L_s = sp.csc_matrix(L_s,dtype=np.float64)
  
    #rate lamda_ik matrix (dgCMatrix)
    if init_param['L_r'] is None: 
        L_r = np.random.gamma(100,scale=0.3/100, size=d*k).reshape(d,k)
    else:
        L_r = init_param['L_r']
    #L_r = sp.csc_matrix(L_r,dtype=np.float64)
    
    if init_param['L2_s'] is None:
        L2_s = np.random.gamma(100,scale=0.3/100, size=d2*k).reshape(d2,k)
    else:
        L2_s = init_param['L2_s']
    #L_s = sp.csc_matrix(L_s,dtype=np.float64)
  
    #rate lamda_ik matrix (dgCMatrix)
    if init_param['L2_r'] is None: 
        L2_r = np.random.gamma(100,scale=0.3/100, size=d2*k).reshape(d2,k)
    else:
        L2_r = init_param['L2_r']
    #L2_r = sp.csc_matrix(L_r,dtype=np.float64)
    #shape kappa_ij matrix
    if init_param['L3_s'] is None:
        tmp = np.copy(C)
        tmp[:,2] = np.random.gamma(100,scale=0.5/100, size= C.shape[0])
        L3_s = np.copy(tmp)
        del(tmp)
    else:
        L3_s = init_param['L3_s']

    ## rate kappa_ij matrix (dgCMatrix)
    if init_param['L3_r'] is None: 
        tmp = np.copy(C)
        tmp[:,2] = np.random.gamma(100,scale=0.5/100, size= C.shape[0])
        L3_r = np.copy(tmp)
        del(tmp)
    else:
        L3_r = init_param['L3_r']

    T3_r = np.repeat(1.0,d2)
    col_sum_c = np.repeat(1,d)
    spC = sp.csc_matrix((C[:,2], (C[:,0], C[:,1])), shape=(d, d2))
    util_sum = spC.sum(axis = 0).A1
    del(spC)

    print('Learning...')    
    #c2pf_cpp(X, C, k, G_s, G_r, L_s, L_r, L2_s, L2_r, L3_s, L3_r, T3_r, col_sum_c, util_sum, iter_max,2.,5.)
    c2pf_cpp(X, C, k, G_s, G_r, L_s, L_r, L2_s, L2_r, L3_s, L3_r, T3_r, col_sum_c, util_sum, iter_max,1e15,1e15)
    c2pf_cpp(X, C, k, G_s, G_r, L_s, L_r, L2_s, L2_r, L3_s, L3_r, T3_r, col_sum_c, util_sum, int(0.2*iter_max),2.,5.)
    print('Learning completed!')
    
    M3 = sp.csc_matrix((np.array(L3_s)[:,2]/np.array(L3_r)[:,2], (np.array(L3_s)[:,0], np.array(L3_s)[:,1])), shape=(d, d2))
    Q = M3*(np.array(L2_s)/np.array(L2_r))
    
    res = {'Z':np.array(G_s)/np.array(G_r),'W':np.array(L_s)/np.array(L_r),'ll': None, 'Q':Q}
    
    return res
    
  
  
#tied-C2PF    
cpdef t_c2pf(Mat &X, int n_X, int d_X, C,int n_C, int d_C, int &k, int &iter_max, init_param = None):
    
    n   = n_X
    d   = d_X
    d2  = d_C

    #Hyper parameter setting
    a  = 0.3
    a_ = 0.3
    c  = 0.3
    c_ = 0.3
    e  = 0.3
    a_t = 100
    b_t = 100/0.5
    b_ = 1
    d_ = 1
    
    #Declare C++ variables
    cdef:
        Mat G_s, G_r, L_s, L_r, L3_s, L3_r
    
    
    
    #Parameters Initialization
    if init_param['G_s'] is None:
        G_s = np.random.gamma(100,scale=0.3/100,size=n*k).reshape(n,k)
    else:
        G_s = init_param['G_s']

    #G_s = sp.csc_matrix(G_s,dtype=np.float64)
  
    # rate gamma_uk matrix
    if init_param['G_r'] is None:
        G_r = np.random.gamma(100,scale=0.3/100, size=n*k).reshape(n,k)
    else:
        G_r = init_param['G_r']
    #G_r = sp.csc_matrix(G_r,dtype=np.float64)
  
    ## shape lamda_ik matrix (dgCMatrix)
    if init_param['L_s'] is None:
        L_s = np.random.gamma(100,scale=0.3/100, size=d*k).reshape(d,k)
                
    else:
        L_s = init_param['L_s']
    #L_s = sp.csc_matrix(L_s,dtype=np.float64)
  
    #rate lamda_ik matrix (dgCMatrix)
    if init_param['L_r'] is None: 
        L_r = np.random.gamma(100,scale=0.3/100, size=d*k).reshape(d,k)
    else:
        L_r = init_param['L_r']
    #L_r = sp.csc_matrix(L_r,dtype=np.float64)
    
    #shape kappa_ij matrix
    if init_param['L3_s'] is None:
        tmp = np.copy(C)
        tmp[:,2] = np.random.gamma(100,scale=0.5/100, size= C.shape[0])
        L3_s = tmp
        del(tmp)
    else:
        L3_s = init_param['L3_s']

    ## rate kappa_ij matrix (dgCMatrix)
    if init_param['L3_r'] is None: 
        tmp = np.copy(C)
        tmp[:,2] = np.random.gamma(100,scale=0.5/100, size= C.shape[0])
        L3_r = tmp
        del(tmp)
    else:
        L3_r = init_param['L3_r']



    T2_r = np.repeat(1.0,d2)
    col_sum_c = np.repeat(1,d)
    spC = sp.csc_matrix((C[:,2], (C[:,0], C[:,1])), shape=(d, d2))
    util_sum = spC.sum(axis = 0).A1
    del(spC)
    
    print('Learning...')
    tc2pf_cpp(X, C, k, G_s, G_r, L_s, L_r, L3_s, L3_r, T2_r, col_sum_c, util_sum, iter_max,1e15,1e15)
    tc2pf_cpp(X, C, k, G_s, G_r, L_s, L_r, L3_s, L3_r, T2_r, col_sum_c, util_sum, int(0.2*iter_max),2.,4.)
    print('Learning completed!')
    
    
    M3 = sp.csc_matrix((np.array(L3_s)[:,2]/np.array(L3_r)[:,2], (np.array(L3_s)[:,0], np.array(L3_s)[:,1])), shape=(d, d2))
    Q = M3*(np.array(L_s)/np.array(L_r))
    
    res = {'Z':np.array(G_s)/np.array(G_r),'W':np.array(L_s)/np.array(L_r),'ll': None, 'Q':Q}
    
    return res  
  
  

#reduced-C2PF
cpdef r_c2pf(Mat &X, int n_X, int d_X, C,int n_C, int d_C, int &k, int &iter_max, init_param = None):
    
    n   = n_X
    d   = d_X
    d2  = d_C

    #Hyper parameter setting
    a  = 0.3
    a_ = 0.3
    a1 = 4.
    c  = 0.3
    c_ = 0.3
    c1 = 3.
    e  = 0.3
    a_t = 100.
    b_t = 100./0.5
    b_ = 1
    d_ = 1
    
    #Declare C++ variables
    cdef:
        Mat G_s, G_r, L2_s, L2_r, L3_s, L3_r
    
    #Parameters Initialization
    if init_param['G_s'] is None:
        G_s = np.random.gamma(100,scale=0.3/100,size=n*k).reshape(n,k)
    else:
        G_s = init_param['G_s']

    #G_s = sp.csc_matrix(G_s,dtype=np.float64)
  
    # rate gamma_uk matrix
    if init_param['G_r'] is None:
        G_r = np.random.gamma(100,scale=0.3/100, size=n*k).reshape(n,k)
    else:
        G_r = init_param['G_r']
    #G_r = sp.csc_matrix(G_r,dtype=np.float64)
    
    if init_param['L2_s'] is None:
        L2_s = np.random.gamma(100,scale=0.3/100, size=d2*k).reshape(d2,k)
    else:
        L2_s = init_param['L2_s']
    #L_s = sp.csc_matrix(L_s,dtype=np.float64)
  
    #rate lamda_ik matrix (dgCMatrix)
    if init_param['L2_r'] is None: 
        L2_r = np.random.gamma(100,scale=0.3/100, size=d2*k).reshape(d2,k)
    else:
        L2_r = init_param['L2_r']
    #L2_r = sp.csc_matrix(L_r,dtype=np.float64)
    #shape kappa_ij matrix
    if init_param['L3_s'] is None: 
        L3_s = np.copy(C)
        L3_s[:,2] = np.random.gamma(100,scale=0.5/100, size= C.shape[0])
    else:
        L3_s = init_param['L3_s']

    ## rate kappa_ij matrix (dgCMatrix)
    if init_param['L3_r'] is None: 
        L3_r = np.copy(C)
        L3_r[:,2] = np.random.gamma(100,scale=0.5/100, size= C.shape[0])
    else:
        L3_r = init_param['L3_r']

    T3_r = np.repeat(1.0,d2)
    col_sum_c = np.repeat(1,d)
    spC = sp.csc_matrix((C[:,2], (C[:,0], C[:,1])), shape=(d, d2))
    util_sum = spC.sum(axis = 0).A1
    del(spC)
    
    print('Learning...')
    rc2pf_cpp(X, C, k, G_s, G_r, L2_s, L2_r, L3_s, L3_r, T3_r, col_sum_c, util_sum, iter_max,1e15,1e15) 
    rc2pf_cpp(X, C, k, G_s, G_r, L2_s, L2_r, L3_s, L3_r, T3_r, col_sum_c, util_sum, int(0.2*iter_max),2.,4.) 
    print('Learning completed!')
    
    M3 = sp.csc_matrix((np.array(L3_s)[:,2]/np.array(L3_r)[:,2], (np.array(L3_s)[:,0], np.array(L3_s)[:,1])), shape=(d, d2))
    Q = M3*(np.array(L2_s)/np.array(L2_r))
    
    res = {'Z':np.array(G_s)/np.array(G_r),'W':Q,'ll': None, 'Q':Q}
    
    return res
