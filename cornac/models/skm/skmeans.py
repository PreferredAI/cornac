# -*- coding: utf-8 -*-
"""
@author: Aghiles Salah
"""

import numpy as np
import scipy.sparse as sp


def skmeans(X, k=5, max_iter=100, tol=1e-6, verbose=True, init_par=None):
    # The Spherical k-means clustering algorithm

    n = X.shape[0]
    print(n)
    print(k)
    # normalize rows of X so as they lie on a unit hypersphere
    X = X.multiply(sp.csc_matrix(1. / (np.sqrt(X.multiply(X).sum(1).A1) + 1e-20)).T)
    if init_par is None:
        par = np.random.randint(k, size=n)
    else:
        par = init_par

    # Initialisation of the classification matrix Z
    Z = sp.csc_matrix((n, k))
    Z[np.arange(n), par] = 1

    change = True
    l_init = -1e1000
    l = []
    iter_ = 0
    while change:
        change = False
        # Update centroids
        MU = Z.T * X
        # project centroids to the unit hypersphere
        MU = MU.multiply(sp.csc_matrix(1. / np.sqrt(MU.multiply(MU).sum(1).A1)).T)
        # MU = sp.csc_matrix(MU)

        # Object Assignements
        Z1 = X * MU.T
        par = Z1.argmax(1).A1  # The object partition in k clusters
        # update the classification matrix
        Z = sp.csc_matrix((n, k))
        Z[np.arange(len(par)), par] = 1

        # Skmeans criteria (likelihood)
        l_t = Z1.multiply(Z).sum()

        if np.abs(l_t - l_init) > tol:
            if verbose:
                print('Iter %i, likelihood: %f' % (iter_, l_t))
            l_init = l_t
            change = True
            l.append(l_t)
            iter_ += 1

    return {"centroids": MU, "partition": par}
