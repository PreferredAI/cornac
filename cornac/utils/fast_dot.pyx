# -*- coding: utf-8 -*-
# cython: language_level=3

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

cimport cython
from cython.parallel import prange


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void fast_dot(float[:] vec, float[:, :] mat, float[:] output) nogil:
    cdef size_t i, j, d0 = mat.shape[0], d1 = mat.shape[1]
    for i in prange(d0):
        for j in prange(d1):
            output[i] += vec[j] * mat[i, j]
