# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
         Quoc-Tuan Truong <tuantq.vnu@gmail.com>
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

    def compute(self, ground_truths, predictions):
        raise NotImplementedError()


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
    def compute(self, ground_truths, predictions, weights=None):
        mae = np.average(np.abs(ground_truths - predictions), axis=0, weights=weights)
        return mae


class MSE(RatingMetric):
    """Mean Squared Error.

    Parameters
    ----------
    name: string, value: 'MSE'
        Name of the measure.
    """
    def __init__(self):
        RatingMetric.__init__(self, name='MSE')

    # Compute MSE
    def compute(self, ground_truths, predictions, weights=None):
        mse = np.average((ground_truths - predictions) ** 2, axis=0, weights=weights)
        return mse


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
    def compute(self, ground_truths, predictions, weights=None):
        mse = np.average((ground_truths - predictions) ** 2, axis=0, weights=weights)
        return np.sqrt(mse)
