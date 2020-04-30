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

from cython.operator cimport dereference as deref, preincrement as inc, predecrement as dec
from libcpp.utility cimport pair
from libcpp.map cimport map as cpp_map

import numpy as np

cimport numpy as np


np.import_array()


cdef class IntFloatDict:

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, np.ndarray[ITYPE_t, ndim=1] keys,
                       np.ndarray[DTYPE_t, ndim=1] values):
        cdef int i
        cdef int size = values.size
        for i in range(size):
            self.my_map[keys[i]] = values[i]

    def __len__(self):
        return self.my_map.size()

    def __getitem__(self, int key):
        cdef cpp_map[ITYPE_t, DTYPE_t].iterator it = self.my_map.find(key)
        if it == self.my_map.end():
            raise KeyError('%i' % key)
        return deref(it).second

    def __setitem__(self, int key, float value):
        self.my_map[key] = value

    def __iter__(self):
        cdef int size = self.my_map.size()
        cdef ITYPE_t [:] keys = np.empty(size, dtype=np.intp)
        cdef DTYPE_t [:] values = np.empty(size, dtype=np.float64)
        self._to_arrays(keys, values)
        cdef int idx
        cdef ITYPE_t key
        cdef DTYPE_t value
        for idx in range(size):
            key = keys[idx]
            value = values[idx]
            yield key, value

    def to_arrays(self):
        """Return the key, value representation of the IntFloatDict
           object.
           Returns
           =======
           keys : ndarray, shape (n_items, ), dtype=int
                The indices of the data points
           values : ndarray, shape (n_items, ), dtype=float
                The values of the data points
        """
        cdef int size = self.my_map.size()
        cdef np.ndarray[ITYPE_t, ndim=1] keys = np.empty(size,
                                                         dtype=np.intp)
        cdef np.ndarray[DTYPE_t, ndim=1] values = np.empty(size,
                                                           dtype=np.float64)
        self._to_arrays(keys, values)
        return keys, values

    cdef _to_arrays(self, ITYPE_t [:] keys, DTYPE_t [:] values):
        cdef cpp_map[ITYPE_t, DTYPE_t].iterator it = self.my_map.begin()
        cdef cpp_map[ITYPE_t, DTYPE_t].iterator end = self.my_map.end()
        cdef int index = 0
        while it != end:
            keys[index] = deref(it).first
            values[index] = deref(it).second
            inc(it)
            index += 1

    def update(self, IntFloatDict other):
        cdef cpp_map[ITYPE_t, DTYPE_t].iterator it = other.my_map.begin()
        cdef cpp_map[ITYPE_t, DTYPE_t].iterator end = other.my_map.end()
        while it != end:
            self.my_map[deref(it).first] = deref(it).second
            inc(it)

    def copy(self):
        cdef IntFloatDict out_obj = IntFloatDict.__new__(IntFloatDict)
        out_obj.my_map = self.my_map
        return out_obj

    def append(self, ITYPE_t key, DTYPE_t value):
        cdef cpp_map[ITYPE_t, DTYPE_t].iterator end = self.my_map.end()
        dec(end)
        cdef pair[ITYPE_t, DTYPE_t] args
        args.first = key
        args.second = value
        self.my_map.insert(end, args)
