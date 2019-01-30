# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

from .result import Result
from .result import SingleModelResult


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
        # think to organize the results first
        self.per_fold_avg[fold] = metric_avg_results

    def _compute_avg_res(self):
        for mt in self.per_fold_avg[0]:
            self.avg[mt] = 0.0
        for f in self.per_fold_avg:
            for mt in self.per_fold_avg[f]:
                self.avg[mt] += self.per_fold_avg[f][mt] / len(self.per_fold_avg)

    def _organize_avg_res(self, model_name, metric_names):
        # global avg
        self.avg = self._get_data_frame(avg_res=self.avg, model_name=model_name, metric_names=metric_names)
        # per_fold avg
        for f in self.per_fold_avg:
            self.per_fold_avg[f] = self._get_data_frame(avg_res=self.per_fold_avg[f], model_name=model_name,
                                                        metric_names=metric_names)


class CVResult(Result):
    """ Cross Validation Result Class

    Parameters
    ----------
    """

    def __init__(self, n_folds, avg_results=None):
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
