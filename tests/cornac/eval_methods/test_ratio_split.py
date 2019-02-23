# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.eval_methods import RatioSplit
from cornac.data import reader
from cornac.models import MF
from cornac.metrics import MAE, Recall

def test_validate_size():
    train_size, val_size, test_size = RatioSplit.validate_size(0.1, 0.2, 10)
    assert 7 == train_size
    assert 1 == val_size
    assert 2 == test_size

    train_size, val_size, test_size = RatioSplit.validate_size(None, 0.5, 10)
    assert 5 == train_size
    assert 0 == val_size
    assert 5 == test_size

    train_size, val_size, test_size = RatioSplit.validate_size(None, None, 10)
    assert 10 == train_size
    assert 0 == val_size
    assert 0 == test_size

    train_size, val_size, test_size = RatioSplit.validate_size(2, 2, 10)
    assert 6 == train_size
    assert 2 == val_size
    assert 2 == test_size

    try:
        RatioSplit.validate_size(-1, 0.2, 10)
    except ValueError:
        assert True

    try:
        RatioSplit.validate_size(1, -0.2, 10)
    except ValueError:
        assert True

    try:
        RatioSplit.validate_size(11, 0.2, 10)
    except ValueError:
        assert True

    try:
        RatioSplit.validate_size(0, 11, 10)
    except ValueError:
        assert True

    try:
        RatioSplit.validate_size(3, 8, 10)
    except ValueError:
        assert True


def test_splits():
    data_file = './tests/data.txt'
    data = reader.read_uir(data_file)

    ratio_split = RatioSplit(data, test_size=0.1, val_size=0.1, seed=123, verbose=True)
    ratio_split.split()
    assert ratio_split._split_ran

    ratio_split.split()


def test_evaluate():
    data_file = './tests/data.txt'
    data = reader.read_uir(data_file)

    ratio_split = RatioSplit(data, exclude_unknowns=True, verbose=True)
    ratio_split.evaluate(MF(), [MAE(), Recall()], user_based=False)

    ratio_split = RatioSplit(data, exclude_unknowns=False, verbose=True)
    ratio_split.evaluate(MF(), [MAE(), Recall()], user_based=False)

    users = []
    items = []
    for u, i, r in data:
        users.append(u)
        items.append(i)
    for u in users:
        for i in items:
            data.append((u, i, 5))

    ratio_split = RatioSplit(data, exclude_unknowns=True, verbose=True)
    ratio_split.evaluate(MF(), [MAE(), Recall()], user_based=True)

    ratio_split = RatioSplit(data, exclude_unknowns=False, verbose=True)
    ratio_split.evaluate(MF(), [MAE(), Recall()], user_based=True)