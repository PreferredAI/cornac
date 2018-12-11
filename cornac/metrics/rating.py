# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np

class RatingMetric:
    """Rating Metric.

    Parameters
    ----------
    name: string,
        Name of the measure.

    type: string, value: 'rating'
        Type of the metric, e.g., "ranking", "rating".
    """

    def __init__(self, name=None):
        self.type = 'rating'
        self.name = name

    def compute(self, data_test, prediction):
        pass


class MAE(RatingMetric):
    """Mean Absolute Error.

    Parameters
    ----------
    name: string, value: 'MAE'
        Name of the measure.
    """

    def __init__(self):
        RatingMetric.__init__(self, name='MAE')

    # Compute MAE
    def compute(self, ground_truths, predictions):
        mae = np.mean(abs(ground_truths - predictions))
        return mae


class RMSE(RatingMetric):
    """Root Mean Squared Error.

    Parameters
    ----------
    name: string, value: 'RMSE'
        Name of the measure.
    """
    def __init__(self):
        RatingMetric.__init__(self, name='RMSE')

    # Compute RMSE
    def compute(self, ground_truths, predictions):
        mse = np.mean((ground_truths - predictions) ** 2)
        return np.sqrt(mse)
