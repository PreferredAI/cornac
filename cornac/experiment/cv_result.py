# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

from .result import Result
from .result import SingleModelResult


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
