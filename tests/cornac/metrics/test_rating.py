# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
import numpy as np
from cornac.metrics.rating import RatingMetric
from cornac.metrics import RMSE
from cornac.metrics import MAE
from cornac.metrics import MSE


class TestRating(unittest.TestCase):

    def test_rating_metric(self):
        metric = RatingMetric()

        self.assertEqual(metric.type, 'rating')
        self.assertIsNone(metric.name)

        try:
            metric.compute()
        except NotImplementedError:
            assert True

    def test_mae(self):
        mae = MAE()

        self.assertEqual(mae.type, 'rating')
        self.assertEqual(mae.name, 'MAE')

        self.assertEqual(0, mae.compute(np.asarray([0]), np.asarray([0])))
        self.assertEqual(1, mae.compute(np.asarray([0, 1]), np.asarray([1, 0])))
        self.assertEqual(2, mae.compute(np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([1, 3])))

    def test_mse(self):
        mse = MSE()

        self.assertEqual(mse.type, 'rating')
        self.assertEqual(mse.name, 'MSE')

        self.assertEqual(0, mse.compute(np.asarray([0]), np.asarray([0])))
        self.assertEqual(1, mse.compute(np.asarray([0, 1]), np.asarray([1, 0])))
        self.assertEqual(4, mse.compute(np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([1, 3])))

    def test_rmse(self):
        rmse = RMSE()

        self.assertEqual(rmse.type, 'rating')
        self.assertEqual(rmse.name, 'RMSE')

        self.assertEqual(0, rmse.compute(np.asarray([0]), np.asarray([0])))
        self.assertEqual(1, rmse.compute(np.asarray([0, 1]), np.asarray([1, 0])))
        self.assertEqual(2, rmse.compute(np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([1, 3])))


if __name__ == '__main__':
    unittest.main()
