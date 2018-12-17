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

    user_based: bool, optional, default: True
        Performance will be averaged based on number of users for rating metrics.
        If `False`, results will be averaged over number of ratings.

    avg_results: DataFrame, default: None
        The average result per model.

    user_results: dictionary, default: {}
        Results per user for each model.
        Result of user u, of metric m, of model d will be user_results[d][m][u]
    """

    def __init__(self, eval_strategy, models, metrics, user_based=True, verbose=False):
        self.eval_strategy = eval_strategy
        self.models = models
        self.metrics = metrics
        self.user_based = user_based
        self.verbose = verbose

        self.res = None
        self.std_result = None
        self.avg_results = None
        self.user_results = {}

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
            organized_metrics[mt.type].append(mt)
            if mt.type == 'ranking':
                ranking_metric_names.append(mt.name)
            elif mt.type == 'rating':
                rating_metric_names.append(mt.name)

        for model in self.models:
            if self.verbose:
                print(model.name)

            model_names.append(model.name)

            metric_avg_results, self.user_results[model.name] = self.eval_strategy.evaluate(model=model,
                                                                                            metrics=organized_metrics,
                                                                                            user_based=self.user_based)

            avg_res = []
            for mt_name in (ranking_metric_names + rating_metric_names):
                avg_res.append(metric_avg_results.get(mt_name, np.nan))

            if self.avg_results is None:
                self.avg_results = np.asarray(avg_res)
            else:
                self.avg_results = np.vstack((self.avg_results, np.asarray(avg_res)))

        # Formatting the results using the Pandas DataFrame
        if len(self.models) == 1:
            self.avg_results = self.avg_results.reshape(1, len(self.metrics))

        self.avg_results = pd.DataFrame(data=self.avg_results, index=model_names,
                                        columns=[*ranking_metric_names, *rating_metric_names])
        print(self.avg_results)