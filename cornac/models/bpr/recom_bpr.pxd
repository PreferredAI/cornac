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

from cython cimport floating, integral
from libcpp cimport bool
from libcpp.vector cimport vector


cdef bool has_non_zero(integral[:], integral[:], integral, integral) nogil


cdef extern from "<boost/random.hpp>" namespace "boost::random":
    cdef cppclass mt19937:
        mt19937(unsigned int)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution(T, T)
        T operator()(mt19937) nogil


cdef class RNGVector(object):
    """ This class creates one c++ rng object per thread, and enables us to randomly sample
    positive/negative items here in a thread safe manner """
    cdef vector[mt19937] rng
    cdef vector[uniform_int_distribution[long]] dist

    cdef inline long generate(self, int thread_id) nogil
