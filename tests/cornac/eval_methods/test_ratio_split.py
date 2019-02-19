# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from cornac.eval_methods import RatioSplit

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
