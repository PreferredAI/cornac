# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
from datetime import datetime

from .result import ExperimentResult
from .result import CVExperimentResult
from ..metrics.rating import RatingMetric
from ..metrics.ranking import RankingMetric
from ..models.recommender import Recommender


class Experiment:
    """ Experiment Class

    Parameters
    ----------
    eval_method: :obj:`<cornac.eval_methods.BaseMethod>`, required
        The evaluation method (e.g., RatioSplit).

    models: array of :obj:`<cornac.models.Recommender>`, required
        A collection of recommender models to evaluate, e.g., [C2PF, HPF, PMF].

    metrics: array of :obj:{`<cornac.metrics.RatingMetric>`, `<cornac.metrics.RankingMetric>`}, required
        A collection of metrics to use to evaluate the recommender models, \
        e.g., [NDCG, MRR, Recall].

    user_based: bool, optional, default: True
        This parameter is only useful if you are considering rating metrics. When True, first the average performance \
        for every user is computed, then the obtained values are averaged to return the final result.
        If `False`, results will be averaged over the number of ratings.

    show_validation: bool, optional, default: True 
        Whether to show the results on validation set (if exists).

    save_dir: str, optional, default: None
        Path to a directory for storing trained models and logs. If None, 
        models will NOT be stored and logs will be saved in the current working directory.

    Attributes
    ----------
    result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment 
        on the test set, initially it is set to None.
    
    val_result: array of :obj:`<cornac.experiment.result.Result>`, default: None
        This attribute contains the results per-model of your experiment
        on the validation set (if exists), initially it is set to None.

    """

    def __init__(
        self,
        eval_method,
        models,
        metrics,
        user_based=True,
        show_validation=True,
        verbose=False,
        save_dir=None,
    ):
        self.eval_method = eval_method
        self.models = self._validate_models(models)
        self.metrics = self._validate_metrics(metrics)
        self.user_based = user_based
        self.show_validation = show_validation
        self.verbose = verbose
        self.save_dir = save_dir
        self.result = None
        self.val_result = None

    @staticmethod
    def _validate_models(input_models):
        if not hasattr(input_models, "__len__"):
            raise ValueError(
                "models have to be an array but {}".format(type(input_models))
            )

        valid_models = []
        for model in input_models:
            if isinstance(model, Recommender):
                valid_models.append(model)
        return valid_models

    @staticmethod
    def _validate_metrics(input_metrics):
        if not hasattr(input_metrics, "__len__"):
            raise ValueError(
                "metrics have to be an array but {}".format(type(input_metrics))
            )

        valid_metrics = []
        for metric in input_metrics:
            if isinstance(metric, RatingMetric) or isinstance(metric, RankingMetric):
                valid_metrics.append(metric)
        return valid_metrics

    def _create_result(self):
        from ..eval_methods.cross_validation import CrossValidation
        from ..eval_methods.propensity_stratified_evaluation import (
            PropensityStratifiedEvaluation,
        )

        if isinstance(self.eval_method, CrossValidation) or isinstance(
            self.eval_method, PropensityStratifiedEvaluation
        ):
            self.result = CVExperimentResult()
        else:
            self.result = ExperimentResult()
            if self.show_validation and self.eval_method.val_set is not None:
                self.val_result = ExperimentResult()

    def run(self):
        """Run the Cornac experiment"""
        self._create_result()

        for model in self.models:
            test_result, val_result = self.eval_method.evaluate(
                model=model,
                metrics=self.metrics,
                user_based=self.user_based,
                show_validation=self.show_validation,
            )

            self.result.append(test_result)
            if self.val_result is not None:
                self.val_result.append(val_result)

            if not isinstance(self.result, CVExperimentResult):
                model.save(self.save_dir)

        output = ""
        if self.val_result is not None:
            output += "\nVALIDATION:\n...\n{}".format(self.val_result)
        output += "\nTEST:\n...\n{}".format(self.result)

        print(output)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        save_dir = "." if self.save_dir is None else self.save_dir
        output_file = os.path.join(save_dir, "CornacExp-{}.log".format(timestamp))
        with open(output_file, "w") as f:
            f.write(output)