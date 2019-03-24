# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
         Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np
import pandas as pd

pd.set_option('precision', 4)


class Result:
    """ Result Class for a single model

    Parameters
    ----------
    """

    def __init__(self, model_name, metric_avg_results, metric_user_results):
        self.model_name = model_name
        self.metric_user_results = metric_user_results
        self.result_df = self.to_df(metric_avg_results)

    def __str__(self):
        self.result_df.index = [self.model_name]
        return self.result_df.__str__()

    @staticmethod
    def to_df(metric_avg_results):
        metric_names = []
        metric_scores = []
        for name, score in metric_avg_results.items():
            metric_names.append(name)
            metric_scores.append(score)

        return pd.DataFrame(data=np.asarray([metric_scores]),
                            columns=np.asarray(metric_names))


class CVResult(list):
    """ Cross Validation Result Class for a single model

    Parameters
    ----------
    """

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def __str__(self):
        return '[{}]\n{}'.format(self.model_name, self.result_df.__str__())

    def organize(self):
        self.result_df = pd.concat([r.result_df for r in self])
        self.result_df.index = ['Fold {}'.format(i + 1) for i in range(self.__len__())]

        self.result_df = self.result_df.T
        mean = self.result_df.mean(axis=1)
        std = self.result_df.std(axis=1)
        self.result_df['Mean'] = mean
        self.result_df['Std'] = std


class ExperimentResult(list):
    """ Result Class

    Parameters
    ----------
    """

    def __str__(self):
        df = pd.concat([r.result_df for r in self])
        df.index = [r.model_name for r in self]
        return df.__str__()


class CVExperimentResult(ExperimentResult):
    """ Cross Validation Result Class

    Parameters
    ----------
    """

    def __str__(self):
        return '\n\n'.join([r.__str__() for r in self])
