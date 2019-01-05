# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np
cimport numpy as np

cimport cython
from libcpp cimport bool

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef sgd(const int[:] rid, const int[:] cid, const double[:] val,
        const int num_users, const int num_items, const int num_factors, const int max_iter,
        double lr, const double reg, const double mu,
        const bool use_bias, const bool early_stop, const bool verbose):
    """Fit the model with SGD
    """

    cdef double[:, :] u_factors = np.random.normal(size=[num_users, num_factors], loc=0., scale=0.01)
    cdef double[:, :] i_factors = np.random.normal(size=[num_items, num_factors], loc=0., scale=0.01)
    cdef double[:] u_biases = np.zeros([num_users])
    cdef double[:] i_biases = np.zeros([num_items])

    cdef double loss = 0
    cdef double last_loss = 0
    cdef double r, r_pred, error, u_f, i_f, delta_loss
    cdef int u, i, factor, j

    for iter in range(1, max_iter + 1):
        last_loss = loss
        loss = 0

        for j in range(val.shape[0]):
            u, i, r = rid[j], cid[j], val[j]

            r_pred = 0
            if use_bias:
                r_pred += mu + u_biases[u] + i_biases[i]
            for factor in range(num_factors):
                r_pred += u_factors[u, factor] * i_factors[i, factor]

            error = r - r_pred
            loss += error * error

            for factor in range(num_factors):
                u_f = u_factors[u, factor]
                i_f = i_factors[i, factor]
                u_factors[u, factor] += lr * (error * i_f - reg * u_f)
                i_factors[i, factor] += lr * (error * u_f - reg * i_f)

        loss = 0.5 * loss

        delta_loss = loss - last_loss
        if early_stop and np.abs(delta_loss) < 1e-5:
            if verbose:
                print('Early stopping, delta_loss = {}'.format(delta_loss))
            break

        if verbose:
            print('Iter {}, loss = {:.2f}'.format(iter, loss))

    if verbose:
        print('Optimization finished!')

    return u_factors, i_factors, u_biases, i_biases