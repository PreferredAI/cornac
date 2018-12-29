# -*- coding: utf-8 -*-
"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

from libcpp.vector cimport vector
import numpy as np
import scipy.sparse as sp

ctypedef vector[vector[double]] Mat
ctypedef vector[double] dVec
ctypedef vector[int] iVec

cdef extern from "cpp_hpf.h":
    void c2pf_cpp(Mat &X, int &g, Mat &G_s, Mat &G_r, Mat &L_s, Mat &L_r, dVec &K_r, dVec &T_r, int maxiter)