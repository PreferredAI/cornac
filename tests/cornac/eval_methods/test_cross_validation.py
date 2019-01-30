# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

from cornac.eval_methods import CrossValidation
from cornac.data import Reader
import numpy as np


def test_partition_data():
    data_file = './tests/data.txt'
    data = Reader.read_uir_triplets(data_file)

    nfolds = 5
    cv = CrossValidation(data=data, n_folds=nfolds)

    ref_set = set(range(nfolds))
    res_set = set(cv.partition)
    fold_sizes = np.unique(cv.partition, return_counts=True)[1]

    assert len(data) == len(cv.partition)
    assert res_set == ref_set
    assert np.all(fold_sizes == 2)


def test_validate_partition():
    data_file = './tests/data.txt'
    data = Reader.read_uir_triplets(data_file)

    nfolds = 5
    cv = CrossValidation(data=data, n_folds=nfolds)

    try:
        cv._validate_partition([0, 0, 1, 1])
    except:
        assert True

    try:
        cv._validate_partition([0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    except:
        assert True
