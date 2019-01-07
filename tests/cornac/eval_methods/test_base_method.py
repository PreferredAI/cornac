# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.eval_methods import BaseMethod
from cornac.data import Reader


def test_init():
    bm = BaseMethod(None, verbose=True)

    assert not bm.exclude_unknowns
    assert 1. == bm.rating_threshold


def test_trainset_none():
    bm = BaseMethod(None, verbose=True)

    try:
        bm.evaluate(None, {}, False)
    except ValueError:
        assert True


def test_testset_none():
    bm = BaseMethod(None, train_set=[], verbose=True)

    try:
        bm.evaluate(None, {}, False)
    except ValueError:
        assert True


def test_from_provided():
    data_file = './tests/data.txt'
    data = Reader.read_uir_triplets(data_file)

    try:
        BaseMethod.from_provided(train_data=None, test_data=None)
    except ValueError:
        assert True

    try:
        BaseMethod.from_provided(train_data=data, test_data=None)
    except ValueError:
        assert True

    bm = BaseMethod.from_provided(train_data=data, test_data=data)

    assert bm.total_users == 10
    assert bm.total_items == 10
