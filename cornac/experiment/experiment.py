# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
         Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from .result import Result
from .result import CVResult

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
        from ..eval_methods.ratio_split import RatioSplit
        from ..eval_methods.cross_validation import CrossValidation
        if isinstance(eval_method, RatioSplit):
            self.results = Result()
        elif isinstance(eval_method, CrossValidation):
            self.results = CVResult(eval_method.n_folds)

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
    def dict_depth(self, d):
        if isinstance(d, dict):
            return 1 + (max(map(self.dict_depth, d.values())) if d else 0)
        return 0

    # modify this function to accommodate several models
    def run(self):
        model_names = []
        metric_names = []
        organized_metrics = {'ranking': [], 'rating': []}

        # Organize metrics into "rating" and "ranking" for efficiency purposes
        for mt in self.metrics:
            organized_metrics[mt.type].append(mt)
            metric_names.append(mt.name)

        for model in self.models:
            if self.verbose:
                print(model.name)

            model_names.append(model.name)
            model_res = self.eval_method.evaluate(model=model, metrics=organized_metrics, user_based=self.user_based)
            model_res._organize_avg_res(model_name=model.name, metric_names=metric_names)
            self.results._add_model_res(res=model_res, model_name=model.name)

        self.results.show()
