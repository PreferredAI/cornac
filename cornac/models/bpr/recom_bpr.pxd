# -*- coding: utf-8 -*-
# cython: language_level=3

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cython cimport floating, integral
from libcpp cimport bool
from libcpp.vector cimport vector


cdef bool has_non_zero(integral[:], integral[:], integral, integral) nogil


cdef extern from "<random>" namespace "std":
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
