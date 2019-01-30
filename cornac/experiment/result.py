# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import numpy as np
import pandas as pd


class SingleModelResult:
    """ Result Class for a single model

    Parameters
    ----------
    """

    def __init__(self, metric_avg_results, metric_user_results=None):
        self.avg = metric_avg_results
        self.per_user = metric_user_results

    def _organize_avg_res(self, model_name, metric_names):
        self.avg = self._get_data_frame(avg_res=self.avg, model_name=model_name, metric_names=metric_names)

    def _get_data_frame(self, avg_res, model_name, metric_names):
        avg_res = [avg_res.get(mt_name, np.nan) for mt_name in metric_names]
        avg_res = np.asarray(avg_res)
        avg_res = avg_res.reshape(1, len(metric_names))
        avg_res = pd.DataFrame(data=avg_res, index=np.asarray([model_name]), columns=np.asarray(metric_names))
        return avg_res


class Result:
    """ Result Class

    Parameters
    ----------
    """
    def __init__(self, avg_results=None):
        self.avg = avg_results
        self.per_user = {}


    def _add_model_res(self, res, model_name):
        self.per_user[model_name] = res.per_user
        if self.avg is None:
            self.avg = res.avg
        else:
            self.avg = self.avg.append(res.avg)

    def show(self):
        print(self.avg)