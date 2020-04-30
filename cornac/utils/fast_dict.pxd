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

from libcpp.map cimport map as cpp_map

cimport numpy as np

ctypedef np.float64_t DTYPE_t

ctypedef np.intp_t ITYPE_t


cdef class IntFloatDict:
    cdef cpp_map[ITYPE_t, DTYPE_t] my_map
    cdef _to_arrays(self, ITYPE_t [:] keys, DTYPE_t [:] values)
