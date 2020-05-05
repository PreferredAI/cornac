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
from libc.math cimport sqrt, fabs
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.stdlib cimport abort, malloc, free
import threading

import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

from tqdm.auto import tqdm



cdef extern from "similarity.h" namespace "cornac_knn" nogil:
    cdef cppclass TopK[Index, Value]:
        TopK(size_t K)
        vector[pair[Value, Index]] results

    cdef cppclass SparseNeighbors[Index, Value]:
        SparseNeighbors(Index max_neighbors)
        void set(Index index, Value weight, Value score)
        void foreach[Function](Function & f)
        vector[Value] weights
        vector[Value] scores


@cython.boundscheck(False)
def compute_similarity(data_mat, unsigned int k=20, unsigned int num_threads=0, verbose=True):
    """ Compute similarity matrix (n_rows x n_rows) of a given data matrix.
    """
    row_mat = data_mat.tocsr()
    col_mat = data_mat.T.tocsr()

    cdef int n_rows = row_mat.shape[0]
    cdef int r, c, i, j
    cdef double w, denom

    cdef int[:] row_indptr = row_mat.indptr, row_indices = row_mat.indices
    cdef double[:] row_data = row_mat.data

    cdef int[:] col_indptr = col_mat.indptr, col_indices = col_mat.indices
    cdef double[:] col_data = col_mat.data
    cdef double[:, :] sim_mat = np.zeros((n_rows, n_rows))
    cdef double * denom1
    cdef double * denom2

    progress = tqdm(total=n_rows, disable=not verbose)
    with nogil, parallel(num_threads=num_threads):
        for r in prange(n_rows, schedule='guided'):
            denom1 = <double *> malloc(sizeof(double) * n_rows)
            denom2 = <double *> malloc(sizeof(double) * n_rows) 
            if denom1 is NULL or denom2 is NULL:
                abort()

            for i in range(n_rows):
                denom1[i] = 0
                denom2[i] = 0

            for i in range(row_indptr[r], row_indptr[r + 1]):
                c, w = row_indices[i], row_data[i]
                for j in range(col_indptr[c], col_indptr[c + 1]):  # neighbors
                    sim_mat[r, col_indices[j]] += col_data[j] * w
                    if w != 0 and col_data[j] != 0:
                        denom1[col_indices[j]] += w * w
                        denom2[col_indices[j]] += col_data[j] * col_data[j]

            for i in range(n_rows):
                if sim_mat[r, i] != 0:
                    denom = sqrt(denom1[i]) * sqrt(denom2[i])
                    sim_mat[r, i] /= denom

            free(denom1)
            free(denom2)

            with gil:
                progress.update(1)
    progress.close()

    sparse_sim_mat = csr_matrix(sim_mat)
    del sim_mat

    return sparse_sim_mat


@cython.boundscheck(False)
def compute_score_single(
    bool user_mode,
    floating[:] sim_arr, 
    int ptr1, 
    int ptr2, 
    int[:] indices, 
    floating[:] data, 
    int k
):
    cdef int max_neighbors = sim_arr.shape[0]
    cdef int nn, j
    cdef double w, s, num, denom, output

    cdef SparseNeighbors[int, double] * neighbours = new SparseNeighbors[int, double](max_neighbors)
    cdef TopK[double, double] * topk = new TopK[double, double](k)
    cdef pair[double, double] result

    for j in range(ptr1, ptr2):
        nn, s = indices[j], data[j]
        if sim_arr[nn] != 0:
            if user_mode:
                neighbours.set(nn, sim_arr[nn], s)
            else:
                neighbours.set(nn, s, sim_arr[nn]) 
        
    topk.results.clear()
    neighbours.foreach(dereference(topk))

    num = 0
    denom = 0
    for result in topk.results:
        w = result.first
        s = result.second
        num = num + w * s
        denom = denom + fabs(w)

    output = num / (denom + 1e-8)

    del topk
    del neighbours

    return output


@cython.boundscheck(False)
def compute_score(
    bool user_mode,
    floating[:] sim_arr,
    int[:] indptr, 
    int[:] indices, 
    floating[:] data,
    unsigned int k, 
    unsigned int num_threads, 
    floating[:] output
):
    cdef int max_neighbors = sim_arr.shape[0]
    cdef int n_items = output.shape[0]
    cdef int nn, i, j
    cdef double w, s, num, denom

    cdef SparseNeighbors[int, double] * neighbours
    cdef TopK[double, double] * topk
    cdef pair[double, double] result

    with nogil, parallel(num_threads=num_threads):
        for i in prange(n_items, schedule='guided'):
            # allocate memory per thread
            neighbours = new SparseNeighbors[int, double](max_neighbors)
            topk = new TopK[double, double](k)

            for j in range(indptr[i], indptr[i + 1]):
                nn, s = indices[j], data[j]
                if sim_arr[nn] != 0:
                    if user_mode:
                        neighbours.set(nn, sim_arr[nn], s)
                    else:
                        neighbours.set(nn, s, sim_arr[nn]) 
                
            topk.results.clear()
            neighbours.foreach(dereference(topk))

            num = 0
            denom = 0
            for result in topk.results:
                w = result.first
                s = result.second
                num = num + w * s
                denom = denom + fabs(w)
            
            output[i] = num / (denom + 1e-8)
                
            del neighbours
            del topk