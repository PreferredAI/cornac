# -*- coding: utf-8 -*-
# cython: language_level=3

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

cimport cython
from cython cimport floating
from cython.parallel import prange

from scipy.linalg.cython_blas cimport sdot, ddot


cdef floating _dot(int n, floating *x, int incx,
                   floating *y, int incy) nogil:
    if floating is float:
        return sdot(&n, x, &incx, y, &incy)
    else:
        return ddot(&n, x, &incx, y, &incy)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void fast_dot(floating[:] vec, floating[:, :] mat, floating[:] output):
    cdef int i, j, d0 = mat.shape[0], d1 = mat.shape[1]
    for i in prange(d0, nogil=True, schedule='static'):
        output[i] += _dot(d1, &vec[0], 1, &mat[i, 0], 1)