# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
         Quoc-Tuan Truong <tuantq.vnu@gmail.com>
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
        self.models = self._validate_models(models)
        self.metrics = self._validate_metrics(metrics)
        self.user_based = user_based
        self.verbose = verbose

        self.res = None
        self.std_result = None
        self.avg_results = []
        self.user_results = {}
        self.fold_avg_results = {}


    @staticmethod
    def _validate_models(input_models):
        if not hasattr(input_models, "__len__"):
            raise ValueError('models have to be an array but {}'.format(type(input_models)))

        from ..models.recommender import Recommender

        valid_models = []
        for model in input_models:
            if isinstance(model, Recommender):
                valid_models.append(model)

        return valid_models


    @staticmethod
    def _validate_metrics(input_metrics):
        if not hasattr(input_metrics, "__len__"):
            raise ValueError('metrics have to be an array but {}'.format(type(input_metrics)))

        from ..metrics.rating import RatingMetric
        from ..metrics.ranking import RankingMetric

        valid_metrics = []
        for metric in input_metrics:
            if isinstance(metric, RatingMetric) or isinstance(metric, RankingMetric):
                valid_metrics.append(metric)

        return valid_metrics
    
    # Check depth of dictionary
    def dict_depth(self,d):
        if isinstance(d, dict):
            return 1 + (max(map(self.dict_depth, d.values())) if d else 0)
        return 0


    # modify this function to accommodate several models
    def run(self):
        model_names = []
        metric_names = []
        organized_metrics = {'ranking':[], 'rating':[]}

        # Organize metrics into "rating" and "ranking" for efficiency purposes
        for mt in self.metrics:
            organized_metrics[mt.type].append(mt)
            metric_names.append(mt.name)

        for model in self.models:
            if self.verbose:
                print(model.name)

            model_names.append(model.name)

            metric_avg_results, self.user_results[model.name] = self.eval_strategy.evaluate(model=model,
                                                                                            metrics=organized_metrics,
                                                                                            user_based=self.user_based)
            
            if self.dict_depth(metric_avg_results) == 1:
                self.avg_results.append([metric_avg_results.get(mt_name, np.nan) for mt_name in metric_names])
                
            elif self.dict_depth(metric_avg_results) == 2:
                for f in metric_avg_results:
                    if f not in self.fold_avg_results:
                        self.fold_avg_results[f] = []
                    self.fold_avg_results[f].append([metric_avg_results[f].get(mt_name, np.nan) for mt_name in metric_names]) 
           
            
        if len(self.fold_avg_results) > 0:
            for f in self.fold_avg_results:
                self.fold_avg_results[f] = pd.DataFrame(data=np.asarray(self.fold_avg_results[f]), index=model_names, columns=metric_names)
                
        if len(self.avg_results) > 0:
            self.avg_results = pd.DataFrame(data=np.asarray(self.avg_results), index=model_names, columns=metric_names)
        
        elif len(self.fold_avg_results) > 0:
            n_folds = 0
            s = 0
            for f in self.fold_avg_results:
                s += self.fold_avg_results[f]
                n_folds += 1
            self.avg_results = s/n_folds
        #print(self.fold_avg_results)