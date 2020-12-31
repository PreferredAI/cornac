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
from libc.math cimport exp
from libcpp cimport bool
from libcpp.string cimport string


ctypedef float FM_FLOAT 


cdef extern from "matrix.h" nogil:

    cppclass DVector[T]:
        DVector() except + 
        DVector(unsigned int) except +

        T get(unsigned int)
        void set(unsigned int, T)

        void init(T)
        void setSize(unsigned int)

        unsigned int dim
        T *value

    cppclass DVectorDouble:
        void init_normal(double mean, double stdev)
        double get(unsigned int)

    cppclass DMatrix[T]:
        DMatrix() except + 
        DMatrix(T) except +

        void init(T)
        void setSize(unsigned int, unsigned int)

        T get(unsigned int, unsigned int)
        void set(unsigned int, unsigned int, T)

    cppclass DMatrixDouble(DMatrix):
        DMatrixDouble() except +
        void set(unsigned int, unsigned int, double)


cdef extern from "fmatrix.h" nogil:

    cppclass sparse_entry[T]:
        sparse_entry() except +
        unsigned int id
        T value

    cppclass sparse_row[T]:
        sparse_row() except +
        sparse_entry[T] *data
        unsigned int size

    cppclass LargeSparseMatrix[T]:
        LargeSparseMatrix() except +

    cppclass LargeSparseMatrixMemory[T]:
        LargeSparseMatrixMemory() except +

        DVector[sparse_row[T]] data

        unsigned int num_cols
        long long num_values
        unsigned int num_rows


cdef extern from "Data.h" nogil:

    cdef cppclass Data:
        Data(int cache_size, bool has_x, bool has_xt) except +
        void load(string filename)
        void load(LargeSparseMatrixMemory[float] &data, DVector[float] &target)
        void debug()

        LargeSparseMatrix[float] *data
        int num_feature
        unsigned int num_cases

        bool verbose

    cppclass DataMetaInfo:
        DataMetaInfo(unsigned int num_attributes) except +

        void debug()

        DVector[unsigned int] attr_group  # attribute_id -> group_id
        unsigned int num_attr_groups
        DVector[unsigned int] num_attr_per_group



cdef extern from "fm_model.h" nogil:

    cppclass fm_model:
        fm_model() except +
        void debug()
        void init()
        double predict(sparse_row[FM_FLOAT] &x)
        # double predict(sparse_row<FM_FLOAT>& x, DVector<double> &sum, DVector<double> &sum_sqr);
        # void saveModel(std::string model_file_path);
        # int loadModel(std::string model_file_path);

        double w0
        DVectorDouble w
        DMatrixDouble v

        # the following values should be set:
        unsigned int num_attribute

        bool k0, k1
        int num_factor

        double reg0
        double regw, regv

        double init_stdev
        double init_mean


cdef extern from "fm_learn.h" nogil:

    cppclass fm_learn:
        fm_learn() except +
        void init()
        double evaluate(Data &data)
        void learn(Data &train, Data &test)
        void debug()

        Data* validation

        DataMetaInfo *meta
        fm_model *fm
        double min_target
        double max_target

        int task


cdef extern from "fm_learn_sgd.h" nogil:

    cppclass fm_learn_sgd(fm_learn):
        fm_learn_sgd() except +

        void learn(Data& train, Data& test)
        
        unsigned int num_iter
        double learn_rate
        DVector[double] learn_rates


cdef extern from "fm_learn_sgd_element.h" nogil:

    cppclass fm_learn_sgd_element(fm_learn_sgd):
        fm_learn_sgd_element() except +


cdef extern from "fm_learn_sgd_element_adapt_reg.h" nogil:

    cppclass fm_learn_sgd_element_adapt_reg(fm_learn_sgd):
        fm_learn_sgd_element_adapt_reg() except +


cdef extern from "fm_learn_mcmc.h" nogil:

    cppclass fm_learn_mcmc(fm_learn):
        fm_learn_mcmc() except + 
        unsigned int num_iter
        unsigned int num_eval_cases

        # Hyperpriors
        double alpha_0, gamma_0, beta_0, mu_0
        double w0_mean_0

        # Priors
        double alpha
        DVector[double] w_mu, w_lambda
        DMatrix[double] v_mu, v_lambda

        # switch between choosing expected values and drawing from distribution
        bool do_sample
        # use the two-level (hierarchical) model (TRUE) or the one-level (FALSE)
        bool do_multilevel

cdef extern from "fm_learn_mcmc_simultaneous.h" nogil:

    cppclass fm_learn_mcmc_simultaneous(fm_learn_mcmc):
        fm_learn_mcmc_simultaneous() except + 