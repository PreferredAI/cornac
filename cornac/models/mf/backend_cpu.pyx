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

import multiprocessing

cimport cython
from cython.parallel import prange
from libcpp cimport bool
from libc.math cimport abs

import numpy as np
cimport numpy as np
from tqdm.auto import trange


ctypedef np.int64_t INT64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_sgd(INT64_t[:] rid, INT64_t[:] cid, float[:] val,
            float[:, :] U, float[:, :] V, 
            float[:] Bu, float[:] Bi,
            float lr, float reg, float mu,
            int max_iter, int num_threads,
            bool use_bias, bool early_stop, bool verbose):
    """Fit the model parameters (U, V, Bu, Bi) with SGD"""
    cdef:
        INT64_t num_ratings = val.shape[0]
        INT64_t u, i, j

        int num_factors = U.shape[1]
        int f

        float loss = 0
        float last_loss = 0
        float r, r_pred, error, u_f, i_f, delta_loss
        

        float * user
        float * item

    progress = trange(max_iter, disable=not verbose)
    for epoch in progress:
        last_loss = loss
        loss = 0

        for j in prange(num_ratings, nogil=True, schedule='static', num_threads=num_threads):
            u, i, r = rid[j], cid[j], val[j]
            user, item = &U[u, 0], &V[i, 0]

            # predict rating
            r_pred = mu + Bu[u] + Bi[i]
            for f in range(num_factors):
                r_pred = r_pred + user[f] * item[f]

            error = r - r_pred
            loss += error * error

            # update factors
            for f in range(num_factors):
                u_f, i_f = user[f], item[f]
                user[f] += lr * (error * i_f - reg * u_f)
                item[f] += lr * (error * u_f - reg * i_f)

            # update biases
            if use_bias:
                Bu[u] += lr * (error - reg * Bu[u])
                Bi[i] += lr * (error - reg * Bi[i])

        loss = 0.5 * loss
        progress.update(1)
        progress.set_postfix({"loss": "%.2f" % loss})

        delta_loss = loss - last_loss
        if early_stop and abs(delta_loss) < 1e-5:
            if verbose:
                print('Early stopping, delta_loss = %.4f' % delta_loss)
            break
    progress.close()

    if verbose:
        print('Optimization finished!')

