# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..ratio_split import RatioSplit


def test_validate_size():
    train_size, val_size, test_size = RatioSplit._validate_sizes(0.1, 0.2, 10)
    assert 7 == train_size
    assert 1 == val_size
    assert 2 == test_size

    train_size, val_size, test_size = RatioSplit._validate_sizes(None, 0.5, 10)
    assert 5 == train_size
    assert 0 == val_size
    assert 5 == test_size

    train_size, val_size, test_size = RatioSplit._validate_sizes(None, None, 10)
    assert 10 == train_size
    assert 0 == val_size
    assert 0 == test_size

    train_size, val_size, test_size = RatioSplit._validate_sizes(2, 2, 10)
    assert 6 == train_size
    assert 2 == val_size
    assert 2 == test_size

    try:
        RatioSplit._validate_sizes(-1, 0.2, 10)
    except ValueError:
        assert True

    try:
        RatioSplit._validate_sizes(11, 0.2, 10)
    except ValueError:
        assert True

    try:
        RatioSplit._validate_sizes(0, 11, 10)
    except ValueError:
        assert True

    try:
        RatioSplit._validate_sizes(3, 8, 10)
    except ValueError:
        assert True