# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""
import numpy as np
from ..utils.util_functions import which_


class RatingMetric:
    """Rating Metric.

    Parameters
    ----------
    type: string, value: 'rating'
        Type of the metric, e.g., "ranking", "rating".
    """

    def __init__(self, name=None):
        self.type = 'rating'

    def compute(self, data_test, prediction):
        pass


class MAE(RatingMetric):
    """Mean Absolute Error.

    Parameters
    ----------
    name: string, value: 'MAE'
        Name of the measure.

    type: string, value: 'rating'
        Type of the metric, e.g., "ranking", "rating".
    """

    def __init__(self):
        RatingMetric.__init__(self, 'MAE')

    # Compute MAE for a single user
    def compute(self, data_test, prediction):
        index_rated = which_(data_test, '>', 0.)
        mae_u = np.sum(abs(data_test[index_rated] - prediction[index_rated])) / len(index_rated)

        return mae_u


class RMSE(RatingMetric):
    """Root Mean Squared Error.

    Parameters
    ----------
    name: string, value: 'RMSE'
        Name of the measure.

    type: string, value: 'prediction'
        Type of the metric, e.g., "ranking", "prediction".
    """

    def __init__(self):
        RatingMetric.__init__(self, 'RMSE')

    # Compute MAE for a single user
    def compute(self, data_test, prediction):
        index_rated = which_(data_test, '>', 0.)
        mse_u = np.sum((data_test[index_rated] - prediction[index_rated]) ** 2) / len(index_rated)

        return np.sqrt(mse_u)
