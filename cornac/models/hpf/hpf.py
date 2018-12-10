# -*- coding: utf-8 -*-
"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import numpy as np
import scipy.sparse as sp
import scipy as sc


# Poisson Factorization
def pf(X, k, max_iter=100, init_param=None):
    ### data preparation
    X = sp.csc_matrix(X, dtype=np.float64)
    M = X.copy()
    M.data = np.full(len(M.data), 1)
    n = X.shape[0]
    d = X.shape[1]
    vtw = np.full(max_iter, 0)
    etp_r = np.full(max_iter, 0)
    etp_c = np.full(max_iter, 0)
    nbIter = max_iter
    eps = 0.000000001

    #### Hyper parameter setting
    a = 0.3
    a_ = 0.3
    c = 0.3
    c_ = 0.3
    b_ = 1.
    d_ = 1.
    # k_s = a_ + k*a
    # t_s = c_ + k*c

    ##### Parameter initialization

    # shape gamma_uk matrix
    if init_param['G_s'] is None:
        G_s = np.random.gamma(a_, scale=b_ / a_, size=n * k).reshape(n, k)
    else:
        G_s = init_param['G_s']

    G_s = sp.csc_matrix(G_s, dtype=np.float64)

    # rate gamma_uk matrix
    if init_param['G_r'] is None:
        G_r = np.random.gamma(a_, scale=b_ / a_, size=n * k).reshape(n, k)
    else:
        G_r = init_param['G_r']

    G_r = sp.csc_matrix(G_r, dtype=np.float64)

    # shape lamda_ik matrix
    if init_param['L_s'] is None:
        L_s = np.random.gamma(c_, scale=d_ / c_, size=d * k).reshape(d, k)

    else:
        L_s = init_param['L_s']

    L_s = sp.csc_matrix(L_s, dtype=np.float64)

    # rate lamda_ik matrix
    if init_param['L_r'] is None:
        L_r = np.random.gamma(c_, scale=d_ / c_, size=d * k).reshape(d, k)
    else:
        L_r = init_param['L_r']
    L_r = sp.csc_matrix(L_r, dtype=np.float64)

    # K_r = a_/b_ + np.sum(G_s/G_r,1)
    # T_r = c_/d_ + np.sum(L_s/L_r,1)

    # Learning
    for iter_ in range(1, max_iter + 1):
        ## Update multinomiale parameter no need to store phi only compute the sufficient stats
        logG_r = G_r.copy()
        logG_r.data = np.log(logG_r.data)
        digaG_s = G_s.copy()
        digaG_s.data = sc.special.digamma(digaG_s.data)

        logL_r = L_r.copy()
        logL_r.data = np.log(logL_r.data)
        digaL_s = L_s.copy()
        digaL_s.data = sc.special.digamma(digaL_s.data)

        Lt = digaG_s - logG_r
        Lt.data = np.exp(Lt.data)
        Lb = digaL_s - logL_r
        Lb.data = np.exp(Lb.data)

        del logG_r
        del digaG_s
        del logL_r
        del digaL_s

        Lt = Lt.todense()
        Lb = Lb.todense()

        # Update user related parameters
        G_s = a + np.multiply(Lt, ((X / (Lt * Lb.T + eps)) * Lb))
        G_r = np.repeat(np.sum(L_s / L_r, 0), n, axis=0) + a  # np.divide(k_s,K_r)

        # K_r = a_/b_ + np.sum(G_s/G_r,1)
        G_s = sp.csc_matrix(G_s)
        G_r = sp.csc_matrix(G_r)

        # Update item related parameters
        L_s = c + np.multiply(Lb, ((X.T / (Lb * Lt.T + eps)) * Lt))
        L_r = np.repeat(np.sum(G_s / G_r, 0), d, axis=0) + c  # np.divide(t_s,T_r)

        # T_r = c_/d_ + np.sum(L_s/L_r,1)
        L_s = sp.csc_matrix(L_s)
        L_r = sp.csc_matrix(L_r)

        Lt = sp.csc_matrix(Lt)
        Lb = sp.csc_matrix(Lb)
    # End of learning

    res = {'Z': G_s / G_r, 'W': L_s / L_r, 'll': vtw}

    return res


# Hierarchical Poisson Factorization
def hpf(X, k, max_iter=100, init_param=None):
    ### data preparation
    X = sp.csc_matrix(X, dtype=np.float64)
    M = X.copy()
    M.data = np.full(len(M.data), 1)
    n = X.shape[0]
    d = X.shape[1]
    vtw = np.full(max_iter, 0)
    etp_r = np.full(max_iter, 0)
    etp_c = np.full(max_iter, 0)
    nbIter = max_iter
    eps = 0.000000001

    #### Hyper parameter setting
    a = 0.3
    a_ = 0.3
    c = 0.3
    c_ = 0.3
    b_ = 1.
    d_ = 1.
    k_s = a_ + k * a
    t_s = c_ + k * c

    ##### Parameter initialization

    # shape gamma_uk matrix
    if init_param['G_s'] is None:
        G_s = np.random.gamma(a_, scale=b_ / a_, size=n * k).reshape(n, k)
    else:
        G_s = init_param['G_s']

    G_s = sp.csc_matrix(G_s, dtype=np.float64)

    # rate gamma_uk matrix
    if init_param['G_r'] is None:
        G_r = np.random.gamma(a_, scale=b_ / a_, size=n * k).reshape(n, k)
    else:
        G_r = init_param['G_r']

    G_r = sp.csc_matrix(G_r, dtype=np.float64)

    # shape lamda_ik matrix
    if init_param['L_s'] is None:
        L_s = np.random.gamma(c_, scale=d_ / c_, size=d * k).reshape(d, k)

    else:
        L_s = init_param['L_s']

    L_s = sp.csc_matrix(L_s, dtype=np.float64)

    # rate lamda_ik matrix
    if init_param['L_r'] is None:
        L_r = np.random.gamma(c_, scale=d_ / c_, size=d * k).reshape(d, k)
    else:
        L_r = init_param['L_r']
    L_r = sp.csc_matrix(L_r, dtype=np.float64)

    K_r = a_ / b_ + np.sum(G_s / G_r, 1)
    T_r = c_ / d_ + np.sum(L_s / L_r, 1)

    # Learning
    for iter_ in range(1, max_iter + 1):
        # Update multinomiale parameter no need to store phi only compute the sufficient stats
        logG_r = G_r.copy()
        logG_r.data = np.log(logG_r.data)
        digaG_s = G_s.copy()
        digaG_s.data = sc.special.digamma(digaG_s.data)

        logL_r = L_r.copy()
        logL_r.data = np.log(logL_r.data)
        digaL_s = L_s.copy()
        digaL_s.data = sc.special.digamma(digaL_s.data)

        Lt = digaG_s - logG_r
        Lt.data = np.exp(Lt.data)
        Lb = digaL_s - logL_r
        Lb.data = np.exp(Lb.data)

        del logG_r
        del digaG_s
        del logL_r
        del digaL_s

        Lt = Lt.todense()
        Lb = Lb.todense()

        # Update user related parameters
        G_s = a + np.multiply(Lt, ((X / (Lt * Lb.T + eps)) * Lb))
        G_r = np.repeat(np.sum(L_s / L_r, 0), n, axis=0) + np.divide(k_s, K_r)

        K_r = a_ / b_ + np.sum(G_s / G_r, 1)
        G_s = sp.csc_matrix(G_s)
        G_r = sp.csc_matrix(G_r)

        # Update item related parameters
        L_s = c + np.multiply(Lb, ((X.T / (Lb * Lt.T + eps)) * Lt))
        L_r = np.repeat(np.sum(G_s / G_r, 0), d, axis=0) + np.divide(t_s, T_r)

        T_r = c_ / d_ + np.sum(L_s / L_r, 1)
        L_s = sp.csc_matrix(L_s)
        L_r = sp.csc_matrix(L_r)

        Lt = sp.csc_matrix(Lt)
        Lb = sp.csc_matrix(Lb)
    # End of learning

    res = {'Z': G_s / G_r, 'W': L_s / L_r, 'll': vtw}

    return res
