# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np

cimport cython
cimport numpy as np

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def sgd(data, num_users, num_items, num_factors, max_iter, learning_rate,
        lambda_reg, global_mean, use_bias, early_stop, verbose):
    """Fit the model with SGD
    """

    cdef np.ndarray[DTYPE_t, ndim=2] u_factors
    cdef np.ndarray[DTYPE_t, ndim=2] i_factors
    cdef np.ndarray[DTYPE_t] u_biases
    cdef np.ndarray[DTYPE_t] i_biases

    u_factors = np.random.normal(size=[num_users, num_factors], loc=0., scale=0.01)
    i_factors = np.random.normal(size=[num_items, num_factors], loc=0., scale=0.01)
    u_biases = np.zeros([num_users], dtype=DTYPE)
    i_biases = np.zeros([num_items], dtype=DTYPE)

    cdef double loss = 0
    cdef double last_loss = 0
    cdef double lr = learning_rate
    cdef double reg = lambda_reg
    cdef double mu = global_mean
    cdef double r, r_pred, error, u_f, i_f, delta_loss
    cdef int u, i, factor
    cdef int k = num_factors

    for iter in range(1, max_iter + 1):
        last_loss = loss
        loss = 0

        for u, i, r in data:
            r_pred = 0
            for factor in range(k):
                r_pred += u_factors[u, factor] * i_factors[i, factor]
            if use_bias:
                r_pred += mu + u_biases[u] + i_biases[i]

            error = r - r_pred
            loss += error * error

            for factor in range(k):
                u_f = u_factors[u, factor]
                i_f = i_factors[i, factor]
                u_factors[u, factor] += lr * (error * i_f - reg * u_f)
                i_factors[i, factor] += lr * (error * u_f - reg * i_f)

        loss = 0.5 * loss

        delta_loss = np.abs(loss - last_loss)
        if early_stop and delta_loss < 1e-5:
            if verbose:
                print('Early stopping, delta_loss = {}'.format(delta_loss))
            break

        if verbose:
            print('Iter {}, loss = {}'.format(iter, loss))

    if verbose:
        print('Optimization finished!')

    return u_factors, i_factors, u_biases, i_biases
