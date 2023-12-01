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
from cython cimport floating, integral
from cython.operator cimport dereference
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free
from libcpp cimport bool
from libcpp.string cimport string

import numpy as np
cimport numpy as np

from .backend_libfm cimport Data
from .backend_libfm cimport DataMetaInfo
from .backend_libfm cimport DVectorDouble
from .backend_libfm cimport sparse_entry
from .backend_libfm cimport sparse_row
from .backend_libfm cimport LargeSparseMatrixMemory

from .backend_libfm cimport fm_model
from .backend_libfm cimport fm_learn
from .backend_libfm cimport fm_learn_sgd
from .backend_libfm cimport fm_learn_sgd_element
from .backend_libfm cimport fm_learn_sgd_element_adapt_reg
from .backend_libfm cimport fm_learn_mcmc
from .backend_libfm cimport fm_learn_mcmc_simultaneous



cdef Data* _prepare_data(long[:] uid, long[:] iid, float[:] val, long num_feature, bool has_t, bool verbose):
        cdef num_values = 2 * len(val)
        cdef num_rows = len(val)

        cdef Data *data = new Data(0, False, has_t)
        data.verbose = verbose

        cdef LargeSparseMatrixMemory[float] *X = new LargeSparseMatrixMemory[float]()
        cdef DVector[float] *target = new DVector[float]()

        X.data.setSize(num_rows)
        target.setSize(num_rows)

        cdef sparse_entry[float] *cache = <sparse_entry[float] *> malloc(num_values * sizeof(sparse_entry[float]))
        if cache is NULL:
            abort()

        cdef unsigned int row_id
        cdef unsigned int cache_id = 0

        for row_id in range(num_rows):
            target.set(row_id, val[row_id])
            X.data.value[row_id].data = &cache[cache_id]
            X.data.value[row_id].size = 0

            # user feature
            cache[cache_id].id = uid[row_id]
            cache[cache_id].value = 1
            cache_id += 1
            X.data.value[row_id].size += 1

            # item feature
            cache[cache_id].id = iid[row_id]
            cache[cache_id].value = 1
            cache_id += 1
            X.data.value[row_id].size += 1 

        assert num_values == cache_id

        X.num_values = num_values
        X.num_cols = num_feature

        data.load(dereference(X), dereference(target))

        return data


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def learn(
    train_set, 
    val_set,
    double w0,
    double[:] w, 
    double[:, :] v,
    unsigned int total_users,
    unsigned int total_items,
    unsigned int k0,
    unsigned int k1,
    unsigned int k2,
    unsigned int max_iter,
    floating learning_rate,
    floating reg0,
    floating reg1,
    floating reg2,
    floating min_rating,
    floating max_rating,
    floating init_stdev,
    str method,
    bool verbose,
):
    cdef unsigned int num_feature = total_users + total_items
    
    (uid, iid, val) = train_set.uir_tuple
    cdef Data *train = _prepare_data(
        uid, 
        iid + total_users, 
        val.astype(np.float32), 
        num_feature, 
        method in ["als", "mcmc"],
        verbose,
    )
    if verbose:
        print("Training data:")
        train.debug()

    cdef Data *validation = NULL
    if val_set is not None:
        (uid, iid, val) = val_set.uir_tuple
        validation = _prepare_data(
            uid, 
            iid + total_users,
            val.astype(np.float32), 
            num_feature, 
            method in ["als", "mcmc"],
            verbose,
        )
        if verbose:
            print("Validation data:")
            validation.debug()
        
    # (1.3) Load meta data
    # (main table)
    cdef DataMetaInfo *meta = new DataMetaInfo(num_feature)
    
    # meta.num_attr_groups = meta_main.num_attr_groups
    meta.num_attr_per_group.setSize(meta.num_attr_groups)
    meta.num_attr_per_group.init(0)
    cdef unsigned int i, j
    for i in range(meta.attr_group.dim):
        j = meta.attr_group.get(i)
        meta.num_attr_per_group.set(j, meta.num_attr_per_group.get(j) + 1)
    
    if verbose:
        meta.debug()

    # (2) Setup the factorization machine
    cdef fm_model fm

    fm.num_attribute = num_feature
    fm.init_stdev = init_stdev
    fm.init_mean = 0.0
    # set the number of dimensions in the factorization
    fm.k0 = k0 != 0
    fm.k1 = k1 != 0
    fm.num_factor = k2
    fm.init()
    # reset the weights
    if k0:
        fm.w0 = w0
    if k1:
        for i in range(num_feature):
            fm.w.set(i, w[i])
    if k2:
        for i in range(k2):
            for j in range(num_feature):
                fm.v.set(i, j, v[i, j]) 

    # (3) Setup the learning method:
    cdef fm_learn *fml

    if method == "sgd":
        fml = new fm_learn_sgd_element()
        (<fm_learn_sgd*>fml).num_iter = max_iter
    elif method == "sgda":
        if val_set is None:
            raise ValueError("'sgda' method requires validation set but None, other methods: ['sgd', 'als', 'mcmc']" )
        
        fml = new fm_learn_sgd_element_adapt_reg()
        (<fm_learn_sgd*>fml).num_iter = max_iter
        (<fm_learn_sgd_element_adapt_reg*>fml).validation = validation
    else: # als or mcmc
        fm.w.init_normal(fm.init_mean, fm.init_stdev)
        fml = new fm_learn_mcmc_simultaneous()
        (<fm_learn_mcmc*>fml).num_iter = max_iter
        (<fm_learn_mcmc*>fml).do_sample = method == "mcmc"
        (<fm_learn_mcmc*>fml).do_multilevel = method == "mcmc"
        if validation != NULL:
            fml.validation = validation
            (<fm_learn_mcmc*>fml).num_eval_cases = validation.num_cases
    
    fml.fm = &fm
    fml.max_target = max_rating
    fml.min_target = min_rating
    fml.meta = meta
    fml.task = 0  # regression

    fml.init()

    # regularization
    fm.reg0 = reg0
    fm.regw = reg1
    fm.regv = reg2
    if method in ["als", "mcmc"]:
        (<fm_learn_mcmc*>fml).w_lambda.init(fm.regw)
        (<fm_learn_mcmc*>fml).v_lambda.init(fm.regv)

    # learning rate
    (<fm_learn_sgd*>fml).learn_rate = learning_rate
    (<fm_learn_sgd*>fml).learn_rates.init(learning_rate)

    if verbose:
        fm.debug()
        fml.debug()

    # () learn
    fml.learn(dereference(train), dereference(validation))

    # store learned weights for prediction
    if k0:
        w0 = fm.w0
    if k1:
        for i in range(num_feature):
            w[i] = fm.w.get(i)
    if k2:
        for i in range(k2):
            for j in range(num_feature):
                v[i, j] = <double>fm.v.get(i, j)
    
    return w0