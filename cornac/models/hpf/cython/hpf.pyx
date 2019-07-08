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

ctypedef vector[vector[double]] Mat
ctypedef vector[double] dVec
ctypedef vector[int] iVec

cdef extern from "cpp_hpf.h":
    void pf_cpp(Mat & X, int & g, Mat & G_s, Mat & G_r, Mat & L_s, Mat & L_r, dVec & K_r, dVec & T_r, int maxiter)
    void hpf_cpp(Mat & X, int & g, Mat & G_s, Mat & G_r, Mat & L_s, Mat & L_r, dVec & K_r, dVec & T_r, int maxiter)




cpdef pf(Mat & X, int n_X, int d_X, int & k, int & iter_max, init_param = None):
    n = n_X
    d = d_X

    #Hyper parameter setting
    a = 0.3
    a_ = 0.3
    a1 = 2.
    c = 0.3
    c_ = 0.3
    c1 = 1.
    b_ = 1
    d_ = 1
    k_s = a + k * a
    t_s = a + k * c

    #Declare C++ variables
    cdef:
        Mat G_s, G_r, L_s, L_r

        #Parameters Initialization
    if init_param['G_s'] is None:
        G_s = np.random.gamma(a_, scale=b_ / a_, size=n * k).reshape(n, k)
    else:
        G_s = init_param['G_s']

    #G_s = sp.csc_matrix(G_s,dtype=np.float64)

    # rate gamma_uk matrix
    if init_param['G_r'] is None:
        G_r = np.random.gamma(a_, scale=b_ / a_, size=n * k).reshape(n, k)
    else:
        G_r = init_param['G_r']
    #G_r = sp.csc_matrix(G_r,dtype=np.float64)

    ## shape lamda_ik matrix (dgCMatrix)
    if init_param['L_s'] is None:
        L_s = np.random.gamma(c_, scale=d_ / c_, size=d * k).reshape(d, k)

    else:
        L_s = init_param['L_s']
    #L_s = sp.csc_matrix(L_s,dtype=np.float64)

    #rate lamda_ik matrix (dgCMatrix)
    if init_param['L_r'] is None:
        L_r = np.random.gamma(c_, scale=d_ / c_, size=d * k).reshape(d, k)
    else:
        L_r = init_param['L_r']
    #L_r = sp.csc_matrix(L_r,dtype=np.float64)  

    #  K_r = c1/b_ + a*rowSums(G_s/G_r)
    #  T_r = c1/d_ + a*rowSums(L_s/L_r)
    K_r = np.repeat(1.0, n)
    T_r = np.repeat(1.0, d)

    print('Learning...')
    pf_cpp(X, k, G_s, G_r, L_s, L_r, K_r, T_r, iter_max)
    print('Learning completed!')

    res = {'Z': np.array(G_s) / np.array(G_r), 'W': np.array(L_s) / np.array(L_r)}

    return res

#Hierarchical Poisson factorization
cpdef hpf(Mat & X, int n_X, int d_X, int & k, int & iter_max, init_param = None):
    n = n_X
    d = d_X

    #Hyper parameter setting
    a = 0.3
    #a_ = 0.3
    a_ = 100.
    a1 = 2.
    c = 0.3
    #c_ = 0.3
    c_ = 100.
    c1 = 1.
    #b_ = 1
    b_ = 0.3
    #d_ = 1
    d_ = 0.3

    #Declare C++ variables
    cdef:
        Mat G_s, G_r, L_s, L_r

        #Parameters Initialization
    if init_param['G_s'] is None:
        G_s = np.random.gamma(a_, scale=b_ / a_, size=n * k).reshape(n, k)
    else:
        G_s = init_param['G_s']

    #G_s = sp.csc_matrix(G_s,dtype=np.float64)

    # rate gamma_uk matrix
    if init_param['G_r'] is None:
        G_r = np.random.gamma(a_, scale=b_ / a_, size=n * k).reshape(n, k)
    else:
        G_r = init_param['G_r']
    #G_r = sp.csc_matrix(G_r,dtype=np.float64)

    ## shape lamda_ik matrix (dgCMatrix)
    if init_param['L_s'] is None:
        L_s = np.random.gamma(c_, scale=d_ / c_, size=d * k).reshape(d, k)

    else:
        L_s = init_param['L_s']
    #L_s = sp.csc_matrix(L_s,dtype=np.float64)

    #rate lamda_ik matrix (dgCMatrix)
    if init_param['L_r'] is None:
        L_r = np.random.gamma(c_, scale=d_ / c_, size=d * k).reshape(d, k)
    else:
        L_r = init_param['L_r']
    #L_r = sp.csc_matrix(L_r,dtype=np.float64)  

    K_r = np.repeat(1.0, n)
    T_r = np.repeat(1.0, d)

    print('Learning...')
    hpf_cpp(X, k, G_s, G_r, L_s, L_r, K_r, T_r, iter_max)
    print('Learning completed!')

    res = {'Z': np.array(G_s) / np.array(G_r), 'W': np.array(L_s) / np.array(L_r)}

    return res
