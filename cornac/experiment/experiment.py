# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
"""

import numpy as np
import pandas as pd


class Experiment:
    """ Experiment Class

    Parameters
    ----------
    eval_strategy: EvaluationStrategy object, required
        The evaluation strategy (e.g., Split).

    models: array of objects Recommender, required
        A collection of recommender models to evaluate, e.g., [C2pf, Hpf, Pmf].

    metrics: array of object metrics, required
        A collection of metrics to use to evaluate the recommender models, \
        e.g., [Ndcg, Mrr, Recall].

    average_result: DataFrame, default: None
        The average result per model.

    result_per_user: dictionary, default: {}
        Results per user for each model.    
    """

    def __init__(self, eval_strategy, models, metrics):
        self.eval_strategy = eval_strategy
        self.models = models
        self.metrics = metrics

        self.res = None
        self.average_result = None
        self.std_result = None
        self.result_per_user = {}

    # modify this function to accomodate several models
    def run(self):

        model_names = []
        ranking_metric_names = []
        rating_metric_names = []
        organized_metrics = {'ranking':[],'rating':[]}

        if not hasattr(self.metrics, "__len__"):
            self.metrics = np.array([self.metrics])  # test whether self.metrics is an array
        if not hasattr(self.models, "__len__"):
            self.models = np.array([self.models])  # test whether self.models is an array

        # Organize metrics into "rating" and "ranking" for efficiency purposes
        for mt in self.metrics:
            if mt.type == 'ranking':
                organized_metrics['ranking'].append(mt)
                ranking_metric_names.append(mt.name)
            else:
                organized_metrics['rating'].append(mt)
                rating_metric_names.append(mt.name)

        for model in self.models:
            print(model.name)
            model_names.append(model.name)
            avg_res, self.result_per_user[model.name] = self.eval_strategy.evaluate(model=model, metrics=organized_metrics)
            if self.average_result is None:
                self.average_result = avg_res
            else:
                self.average_result = np.vstack((self.average_result, avg_res))

        # Formatting the results using the Pandas DataFrame
        if len(self.models) == 1:
            self.average_result = self.average_result.reshape(1, len(self.metrics))
        resAvg_dataFrame = pd.DataFrame(data=self.average_result, index=model_names, columns=[*ranking_metric_names,*rating_metric_names])
        self.average_result = resAvg_dataFrame
        ##Metrics, take into account the metrics specified by the user
        del (resAvg_dataFrame)
