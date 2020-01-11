# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np


class RatingMetric:
    """Rating Metric.

    Attributes
    ----------
    name: string,
        Name of the measure.

    type: string, value: 'rating'
        Type of the metric, e.g., "ranking", "rating".

    """

    def __init__(self, name=None, higher_better=False):
        self.type = 'rating'
        self.name = name
        self.higher_better = higher_better

    def compute(self, **kwargs):
        raise NotImplementedError()


class MAE(RatingMetric):
    """Mean Absolute Error.

    Attributes
    ----------
    name: string, value: 'MAE'
        Name of the measure.

    """

    def __init__(self):
        RatingMetric.__init__(self, name='MAE')

    def compute(self, gt_ratings, pd_ratings, weights=None, **kwargs):
        """Compute Mean Absolute Error.

        Parameters
        ----------
        gt_ratings: Numpy array
            Ground-truth rating values.

        pd_ratings: Numpy array
            Predicted rating values.

        weights: Numpy array, optional, default: None
            Weights for rating values.

        **kwargs: For compatibility

        Returns
        -------
        mae: A scalar.
            Mean Absolute Error.

        """
        mae = np.average(np.abs(gt_ratings - pd_ratings), axis=0, weights=weights)
        return mae


class MSE(RatingMetric):
    """Mean Squared Error.

    Attributes
    ----------
    name: string, value: 'MSE'
        Name of the measure.

    """

    def __init__(self):
        RatingMetric.__init__(self, name='MSE')

    def compute(self, gt_ratings, pd_ratings, weights=None, **kwargs):
        """Compute Mean Squared Error.

        Parameters
        ----------
        gt_ratings: Numpy array
            Ground-truth rating values.

        pd_ratings: Numpy array
            Predicted rating values.

        weights: Numpy array, optional, default: None
            Weights for rating values.

        **kwargs: For compatibility

        Returns
        -------
        mse: A scalar.
            Mean Squared Error.

        """
        mse = np.average((gt_ratings - pd_ratings) ** 2, axis=0, weights=weights)
        return mse


class RMSE(RatingMetric):
    """Root Mean Squared Error.

    Attributes
    ----------
    name: string, value: 'RMSE'
        Name of the measure.

    """

    def __init__(self):
        RatingMetric.__init__(self, name='RMSE')

    def compute(self, gt_ratings, pd_ratings, weights=None, **kwargs):
        """Compute Root Mean Squared Error.

        Parameters
        ----------
        gt_ratings: Numpy array
            Ground-truth rating values.

        pd_ratings: Numpy array
            Predicted rating values.

        weights: Numpy array, optional, default: None
            Weights for rating values.

        **kwargs: For compatibility

        Returns
        -------
        rmse: A scalar.
            Root Mean Squared Error.

        """
        mse = np.average((gt_ratings - pd_ratings) ** 2, axis=0, weights=weights)
        return np.sqrt(mse)
