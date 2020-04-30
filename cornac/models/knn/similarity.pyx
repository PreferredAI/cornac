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

import cython
from cython cimport floating, integral
from cython.operator import dereference
from cython.parallel import parallel, prange
from libc.math cimport sqrt
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.utility cimport pair
import threading

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

from tqdm.auto import tqdm



cdef extern from "similarity.h" namespace "cornac_knn" nogil:
    cdef cppclass TopK[Index, Value]:
        TopK(size_t K)
        vector[pair[Value, Index]] results

    cdef cppclass SparseMatrixMultiplier[Index, Value]:
        SparseMatrixMultiplier(Index n_rows)
        void add(Index index, Value value)
        void foreach[Function](Function & f)
        vector[Value] sums


@cython.boundscheck(False)
def compute_similarity(data_mat, unsigned int k=20, unsigned int num_threads=0, verbose=True):
    """ Compute similarity matrix (n_rows x n_rows) of a given data matrix.
    """
    row_mat = data_mat.tocsr()
    col_mat = data_mat.T.tocsr()

    cdef int n_rows = row_mat.shape[0]
    cdef int r, c, i, j
    cdef double w

    cdef int[:] row_indptr = row_mat.indptr, row_indices = row_mat.indices
    cdef double[:] row_data = row_mat.data

    cdef int[:] col_indptr = col_mat.indptr, col_indices = col_mat.indices
    cdef double[:] col_data = col_mat.data

    cdef SparseMatrixMultiplier[int, double] * neighbours
    cdef TopK[int, double] * topk
    cdef pair[double, int] result

    # holds triples of output similarity matrix
    cdef double[:] values = np.zeros(n_rows * k)
    cdef long[:] rows = np.zeros(n_rows * k, dtype=int)
    cdef long[:] cols = np.zeros(n_rows * k, dtype=int)

    progress = tqdm(total=n_rows, disable=not verbose)
    with nogil, parallel(num_threads=num_threads):
        # allocate memory per thread
        neighbours = new SparseMatrixMultiplier[int, double](n_rows)
        topk = new TopK[int, double](k)

        try:
            for r in prange(n_rows, schedule='guided'):
                for i in range(row_indptr[r], row_indptr[r + 1]):
                    c = row_indices[i]
                    w = row_data[i]

                    for j in range(col_indptr[c], col_indptr[c + 1]):
                        neighbours.add(col_indices[j], col_data[j] * w)

                topk.results.clear()
                neighbours.foreach(dereference(topk))

                i = k * r
                for result in topk.results:
                    rows[i] = r
                    cols[i] = result.second
                    values[i] = result.first
                    i = i + 1
                with gil:
                    progress.update(1)

        finally:
            del neighbours
            del topk
    progress.close()

    return csr_matrix((values, (rows, cols)), shape=(n_rows, n_rows))