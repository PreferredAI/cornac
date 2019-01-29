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
        #self.avg = [self.avg.get(mt_name, np.nan) for mt_name in metric_names]
        #self.avg = np.asarray(self.avg)
        #self.avg = self.avg.reshape(1, len(metric_names))
        #self.avg = pd.DataFrame(data=self.avg, index=np.asarray([model_name]), columns=np.asarray(metric_names))
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
    def __init__(self, per_model_results = {}, avg_results=None):
        #self.per_model = per_model_results
        self.avg = avg_results
        self.per_user = {}


    def _add_model_res(self, res, model_name):
        #self.per_model[model_name] = res
        self.per_user[model_name] = res.per_user
        if self.avg is None:
            self.avg = res.avg
        else:
            self.avg = self.avg.append(res.avg)

    def show(self):
        print(self.avg)




class CVSingleModelResult(SingleModelResult):
    """ Cross Validation Result Class for a single model

    Parameters
    ----------
    """

    def __init__(self, metric_avg_results=None):
        self.avg = metric_avg_results
        self.per_fold_avg = {}
        self.avg = {}

    def _add_fold_res(self, fold, metric_avg_results):
        #think to organize the results first
        self.per_fold_avg[fold] = metric_avg_results

    def _compute_avg_res(self):
        for mt in self.per_fold_avg[0]:
            self.avg[mt] = 0.0
        for f in self.per_fold_avg:
            for mt in self.per_fold_avg[f]:
                self.avg[mt] += self.per_fold_avg[f][mt]/len(self.per_fold_avg)

    def _organize_avg_res(self, model_name, metric_names):
        # global avg
        self.avg = self._get_data_frame(avg_res=self.avg, model_name=model_name, metric_names=metric_names)
        # per_fold avg
        for f in self.per_fold_avg:
            self.per_fold_avg[f] = self._get_data_frame(avg_res=self.per_fold_avg[f], model_name= model_name,
                                                        metric_names=metric_names)


class CVResult(Result):
    """ Cross Validation Result Class

    Parameters
    ----------
    """
    def __init__(self, per_model_results = {}, n_folds, avg_results=None):
        #self.per_model = per_model_results
        self.avg = avg_results
        self.per_fold_avg = {}
        for f in range(n_folds):
            self.per_fold_avg[f] = None


    def _add_model_res(self, res, model_name):
        if self.avg is None:
            self.avg = res.avg
        else:
            self.avg = self.avg.append(res.avg)
        for f in res.per_fold_avg:
            if self.per_fold_avg[f] is None:
                self.per_fold_avg[f] = res.per_fold_avg[f]
            else:
                self.per_fold_avg[f] = self.per_fold_avg[f].append(res.per_fold_avg[f])
