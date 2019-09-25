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


# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

cimport cython
from cython cimport floating
from libc.math cimport fabs, sqrt

cimport numpy as np
import numpy as np

ctypedef fused integral:
    int
    long long

def inplace_csr_row_normalize_l1(X):
    """Inplace row normalize using the l1 norm"""
    _inplace_csr_row_normalize_l1(X.data, X.shape, X.indices, X.indptr)

def _inplace_csr_row_normalize_l1(np.ndarray[floating, ndim=1] X_data,
                                  shape,
                                  np.ndarray[integral, ndim=1] X_indices,
                                  np.ndarray[integral, ndim=1] X_indptr):
    cdef unsigned long long n_samples = shape[0]
    cdef unsigned long long n_features = shape[1]

    # the column indices for row i are stored in:
    #    indices[indptr[i]:indices[i+1]]
    # and their corresponding values are stored in:
    #    data[indptr[i]:indptr[i+1]]
    cdef np.npy_intp i, j
    cdef double sum_

    for i in range(n_samples):
        sum_ = 0.0

        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += fabs(X_data[j])

        if sum_ == 0.0:
            # do not normalize empty rows (can happen if CSR is not pruned
            # correctly)
            continue

        for j in range(X_indptr[i], X_indptr[i + 1]):
            X_data[j] /= sum_

def inplace_csr_row_normalize_l2(X):
    """Inplace row normalize using the l2 norm"""
    _inplace_csr_row_normalize_l2(X.data, X.shape, X.indices, X.indptr)

def _inplace_csr_row_normalize_l2(np.ndarray[floating, ndim=1] X_data,
                                  shape,
                                  np.ndarray[integral, ndim=1] X_indices,
                                  np.ndarray[integral, ndim=1] X_indptr):
    cdef integral n_samples = shape[0]
    cdef integral n_features = shape[1]

    cdef np.npy_intp i, j
    cdef double sum_

    for i in range(n_samples):
        sum_ = 0.0

        for j in range(X_indptr[i], X_indptr[i + 1]):
            sum_ += (X_data[j] * X_data[j])

        if sum_ == 0.0:
            # do not normalize empty rows (can happen if CSR is not pruned
            # correctly)
            continue

        sum_ = sqrt(sum_)

        for j in range(X_indptr[i], X_indptr[i + 1]):
            X_data[j] /= sum_
