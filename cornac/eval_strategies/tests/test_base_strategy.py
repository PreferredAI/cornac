# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..base_strategy import BaseStrategy


def test_init():
    bs = BaseStrategy(None, verbose=True)

    assert not bs.exclude_unknowns
    assert 1. == bs.rating_threshold


def test_trainset_none():
    bs = BaseStrategy(None, verbose=True)

    try:
        bs.evaluate(None, {}, False)
    except ValueError:
        assert True


def test_testset_none():
    bs = BaseStrategy(None, train_set=[], verbose=True)

    try:
        bs.evaluate(None, {}, False)
    except ValueError:
        assert True


