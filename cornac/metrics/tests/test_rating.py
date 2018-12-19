# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np


def test_rating_metric():
    from ..rating import RatingMetric
    metric = RatingMetric()

    assert metric.type == 'rating'
    assert metric.name is None

    try:
        metric.compute(None, None)
    except NotImplementedError:
        assert True


def test_mae():
    from ..rating import MAE
    mae = MAE()

    assert mae.type == 'rating'
    assert mae.name == 'MAE'

    assert 0 == mae.compute(np.asarray([0]), np.asarray([0]))
    assert 1 == mae.compute(np.asarray([0, 1]), np.asarray([1, 0]))
    assert 2 == mae.compute(np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([1, 3]))


def test_mse():
    from ..rating import MSE
    mse = MSE()

    assert mse.type == 'rating'
    assert mse.name == 'MSE'

    assert 0 == mse.compute(np.asarray([0]), np.asarray([0]))
    assert 1 == mse.compute(np.asarray([0, 1]), np.asarray([1, 0]))
    assert 4 == mse.compute(np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([1, 3]))


def test_rmse():
    from ..rating import RMSE
    rmse = RMSE()

    assert rmse.type == 'rating'
    assert rmse.name == 'RMSE'

    assert 0 == rmse.compute(np.asarray([0]), np.asarray([0]))
    assert 1 == rmse.compute(np.asarray([0, 1]), np.asarray([1, 0]))
    assert 2 == rmse.compute(np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([1, 3]))