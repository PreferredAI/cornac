# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
         Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from .result import ExperimentResult
from .result import CVExperimentResult
from ..metrics.rating import RatingMetric
from ..metrics.ranking import RankingMetric
from ..models.recommender import Recommender


class Experiment:
    """ Experiment Class

    Parameters
    ----------
    eval_method: BaseMethod object, required
        The evaluation method (e.g., RatioSplit).

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

    def __init__(self, eval_method, models, metrics, user_based=True, verbose=False):
        self.eval_method = eval_method
        self.models = self._validate_models(models)
        self.metrics = self._validate_metrics(metrics)
        self.user_based = user_based
        self.verbose = verbose
        self.result = None

    @staticmethod
    def _validate_models(input_models):
        if not hasattr(input_models, "__len__"):
            raise ValueError('models have to be an array but {}'.format(type(input_models)))

        valid_models = []
        for model in input_models:
            if isinstance(model, Recommender):
                valid_models.append(model)
        return valid_models

    @staticmethod
    def _validate_metrics(input_metrics):
        if not hasattr(input_metrics, "__len__"):
            raise ValueError('metrics have to be an array but {}'.format(type(input_metrics)))

        valid_metrics = []
        for metric in input_metrics:
            if isinstance(metric, RatingMetric) or isinstance(metric, RankingMetric):
                valid_metrics.append(metric)
        return valid_metrics

    def _create_result(self):
        from ..eval_methods.cross_validation import CrossValidation
        if isinstance(self.eval_method, CrossValidation):
            self.result = CVExperimentResult()
        else:
            self.result = ExperimentResult()

    def run(self):
        self._create_result()
        for model in self.models:
            model_result = self.eval_method.evaluate(model=model,
                                                     metrics=self.metrics,
                                                     user_based=self.user_based)
            self.result.append(model_result)
        print('\n{}'.format(self.result))