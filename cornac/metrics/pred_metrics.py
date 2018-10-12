# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""
import numpy as np
from ..utils.util_functions import which_



class Mae:
    """Mean Absolute Error.

    Parameters
    ----------
    name: string, value: 'MAE'
        Name of the measure.

    type: string, value: 'prediction'
        Type of the metric, e.g., "ranking", "prediction".
    """

    def __init__(self):
        self.name = 'MAE'
        self.type = 'prediction'

    #Compute MAE for a single user
    def compute(self,data_test,prediction):
        index_rated = which_(data_test,'>',0.)
        mae_u = np.sum(abs(data_test[index_rated] - prediction[index_rated]))/len(index_rated)

        return mae_u

    
class Rmse:
    """Root Mean Squared Error.

    Parameters
    ----------
    name: string, value: 'RMSE'
        Name of the measure.

    type: string, value: 'prediction'
        Type of the metric, e.g., "ranking", "prediction".
    """

    def __init__(self):
        self.name = 'RMSE'
        self.type = 'prediction'

    #Compute MAE for a single user
    def compute(self,data_test,prediction):
        index_rated = which_(data_test,'>',0.)
        mse_u = np.sum((data_test[index_rated] - prediction[index_rated])**2)/len(index_rated)

        return np.sqrt(mse_u)