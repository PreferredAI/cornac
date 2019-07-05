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

# cython: language_level=3

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
