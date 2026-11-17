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


import numpy as np
from itertools import product

from .models import Recommender, NextItemRecommender
from .metrics import RatingMetric, RankingMetric
from .eval_methods import rating_eval, ranking_eval
from .eval_methods.next_item_evaluation import ranking_eval as next_item_ranking_eval
from .utils import get_rng


__all__ = ["Discrete", "Continuous", "GridSearch", "RandomSearch", "OptunaSearch"]


class SearchDomain(object):
    """Domain of a parameter to search on.
    
    Parameters
    ----------------
    name: str, required
        Name of the parameter.

    """

    def __init__(self, name):
        self.name = name

    def _sample(self, rng):
        """Sample a value of parameter used for RandomSearch"""
        raise NotImplementedError()


class Discrete(SearchDomain):
    """Domain of a parameter with a set of discrete values.
    
    Parameters
    ----------------
    name: str, required
        Name of the parameter.
        
    values: list, required
        List of values to be searched.

    """

    def __init__(self, name, values):
        super().__init__(name=name)
        self.values = values

    def _sample(self, rng):
        """Sample a value of parameter used for RandomSearch"""
        return rng.choice(self.values)


class Continuous(SearchDomain):
    """Domain of a parameter with continuous values within a range of [low, high).
    
    Parameters
    ----------------
    name: str, required
        Name of the parameter.
        
    low: float, default: 0.0
        Lower bound of the searched values (included).

    high: float, default: 1.0
        Upper bound of the searched values (excluded).

    log: bool, default: False
        If True, values are sampled uniformly on a log scale.
        Requires `low` > 0.

    """

    def __init__(self, name, low=0.0, high=1.0, log=False):
        super().__init__(name=name)
        self.low = low
        self.high = high
        self.log = log

    def _sample(self, rng):
        """Sample a value of parameter used for RandomSearch"""
        if self.log:
            return np.exp(rng.uniform(low=np.log(self.low), high=np.log(self.high)))
        return rng.uniform(low=self.low, high=self.high)


class BaseSearch(Recommender):
    """Base class for doing parameter search.
    
    Parameters
    ----------------
    model: :obj:`cornac.models.Recommender`, required
        Base recommender model to be tuned.

    space: list, required
        Parameter space to be searched on.
        It's a list of :obj:`cornac.hyperopt.SearchDomain`.
    
    metric: :obj:`cornac.metrics.RatingMetric` or :obj:`cornac.metrics.RankingMetric`, required
        Scoring metric to measure the performance and rank the parameter settings.

    eval_method: :obj:`cornac.eval_methods.BaseMethod`, required
        Evaluation method is being used. 
        
    name: str, default: 'BaseSearch'
        The name of the searching strategy.
        
    """

    def __init__(self, model, space, metric, eval_method, name="BaseSearch"):
        super().__init__(name=name, verbose=model.verbose)
        self.model = model
        self.space = sorted(space, key=lambda x: x.name)  # for reproducibility
        self.metric = metric
        self.eval_method = eval_method

    def _build_param_set(self):
        """Generate searching points"""
        raise NotImplementedError()

    def _evaluate(self, model, train_set, val_set):
        """Score a fitted model on the validation set"""
        if isinstance(self.metric, RatingMetric):
            return rating_eval(model, [self.metric], val_set)[0][0]
        if isinstance(model, NextItemRecommender):
            return next_item_ranking_eval(
                model,
                [self.metric],
                train_set,
                val_set,
                exclude_unknowns=self.eval_method.exclude_unknowns,
                mode=self.eval_method.mode,
                verbose=False,
            )[0][0]
        return ranking_eval(
            model,
            [self.metric],
            train_set,
            val_set,
            rating_threshold=self.eval_method.rating_threshold,
            exclude_unknowns=self.eval_method.exclude_unknowns,
            verbose=False,
        )[0][0]

    def fit(self, train_set, val_set=None):
        """Doing hyper-parameter search"""
        assert val_set is not None
        Recommender.fit(self, train_set, val_set)

        param_set = self._build_param_set()
        compare_op = np.greater if self.metric.higher_better else np.less
        self.best_score = -np.inf if self.metric.higher_better else np.inf
        self.best_model = None
        self.best_params = None

        # this can be parallelized if needed
        # keep it simple because multimodal algorithms are usually resource-hungry
        for params in param_set:
            if self.verbose:
                print("Evaluating: {}".format(params))

            model = self.model.clone(params).fit(train_set, val_set)
            score = self._evaluate(model, train_set, val_set)

            if compare_op(score, self.best_score):
                self.best_score = score
                self.best_model = model
                self.best_params = params

            del model

        if self.verbose:
            print("Best parameter settings: {}".format(self.best_params))
            print("{} = {:.4f}".format(self.metric.name, self.best_score))

        return self

    def transform(self, test_set):
        """Delegate test-set transformation to the best searched model."""
        return self.best_model.transform(test_set)

    def score(self, user_idx, *args, **kwargs):
        """Scoring using the best searched model"""
        return self.best_model.score(user_idx, *args, **kwargs)

    def rank(self, user_idx, item_indices=None, k=-1, **kwargs):
        """Ranking using the best searched model"""
        return self.best_model.rank(user_idx, item_indices, k, **kwargs)


class GridSearch(BaseSearch):
    """Parameter searching on a grid.
    
    Parameters
    ----------------
    model: :obj:`cornac.models.Recommender`, required
        Base recommender model to be tuned.

    space: list, required
        Parameter space to be searched on.
        It's a list of :obj:`cornac.hyperopt.SearchDomain`.
    
    metric: :obj:`cornac.metrics.RatingMetric` or :obj:`cornac.metrics.RankingMetric`, required
        Scoring metric to measure the performance and rank the parameter settings.

    eval_method: :obj:`cornac.eval_methods.BaseMethod`, required
        Evaluation method is being used. 
        
    """

    def __init__(self, model, space, metric, eval_method):
        super().__init__(
            model,
            self._validate(space),
            metric,
            eval_method,
            name="GridSearch_{}".format(model.name),
        )

    @staticmethod
    def _validate(space):
        """GridSearch only accepts Discrete search domain"""
        for domain in space:
            if isinstance(domain, Discrete):
                continue

            raise ValueError(
                "GridSearch only supports Discrete domain but {} is not!\n\
                    Please consider using RandomSearch instead.".format(
                    domain.name
                )
            )

        return space

    def _build_param_set(self):
        """Generate searching points"""
        param_set = []
        keys = [d.name for d in self.space]
        for params in product(*[sorted(d.values) for d in self.space]):
            param_set.append(dict(zip(keys, params)))
        return param_set


class RandomSearch(BaseSearch):
    """Parameter searching with random strategy.
    
    Parameters
    ----------------
    model: :obj:`cornac.models.Recommender`, required
        Base recommender model to be tuned.

    space: list, required
        Parameter space to be searched on.
        It's a list of :obj:`cornac.hyperopt.SearchDomain`.
    
    metric: :obj:`cornac.metrics.RatingMetric` or :obj:`cornac.metrics.RankingMetric`, required
        Scoring metric to measure the performance and rank the parameter settings.

    eval_method: :obj:`cornac.eval_methods.BaseMethod`, required
        Evaluation method is being used. 

    n_trails: int, default: 10
        Number of trails for random search.

    """

    def __init__(self, model, space, metric, eval_method, n_trails=10):
        super().__init__(
            model, space, metric, eval_method, name="RandomSearch_{}".format(model.name)
        )
        self.n_trails = n_trails

    def _build_param_set(self):
        """Generate searching points"""
        param_set = []
        keys = [d.name for d in self.space]
        rng = get_rng(getattr(self.model, "seed", None))
        while len(param_set) < self.n_trails:
            params = [d._sample(rng) for d in self.space]
            param_set.append(dict(zip(keys, params)))
        return param_set


class OptunaSearch(BaseSearch):
    """Parameter searching with `Optuna <https://optuna.org>`_.

    Parameters
    ----------------
    model: :obj:`cornac.models.Recommender`, required
        Base recommender model to be tuned.

    space: list, required
        Parameter space to be searched on.
        It's a list of :obj:`cornac.hyperopt.SearchDomain`.

    metric: :obj:`cornac.metrics.RatingMetric` or :obj:`cornac.metrics.RankingMetric`, required
        Scoring metric to measure the performance and rank the parameter settings.

    eval_method: :obj:`cornac.eval_methods.BaseMethod`, required
        Evaluation method is being used.

    n_trials: int, default: 10
        Number of trials to run. When a stored study is resumed,
        they are added to the trials it already has.

    sampler: :obj:`optuna.samplers.BaseSampler`, default: None
        Sampler proposing the parameter settings. By default, it is
        a `TPESampler` seeded with the seed of the model.

    pruner: :obj:`optuna.pruners.BasePruner`, default: None
        Pruner stopping unpromising trials, `MedianPruner` by default.
        A trial can be pruned only if the model checks for early stopping
        on the validation set after each epoch.

    storage: str or :obj:`optuna.storages.BaseStorage`, default: None
        Where to store the study, e.g. "sqlite:///study.db".
        By default, the study is kept in memory.

    study_name: str, default: None
        Name of the study. Fitting again with the same `storage` and
        `study_name` resumes the study.

    Attributes
    ----------
    study: :obj:`optuna.study.Study`
        The study with the history of all trials, available after fitting.

    """

    def __init__(
        self,
        model,
        space,
        metric,
        eval_method,
        n_trials=10,
        sampler=None,
        pruner=None,
        storage=None,
        study_name=None,
    ):
        super().__init__(
            model, space, metric, eval_method, name="OptunaSearch_{}".format(model.name)
        )
        self.n_trials = n_trials
        self.sampler = sampler
        self.pruner = pruner
        self.storage = storage
        self.study_name = study_name

    def _fit_trial(self, trial, params, train_set, val_set):
        """Fit a model, reporting its monitored value to the trial after each epoch"""
        import optuna

        model = self.model.clone(params)
        monitor_value = model.monitor_value
        # monitored values are larger-is-better, pruners follow the study direction
        sign = 1 if self.metric.higher_better else -1

        def report(train_set, val_set):
            value = monitor_value(train_set, val_set)
            if value is not None:
                trial.report(sign * value, model.current_epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return value

        model.monitor_value = report
        try:
            return model.fit(train_set, val_set)
        finally:
            del model.monitor_value

    def fit(self, train_set, val_set=None):
        """Doing hyper-parameter search"""
        import optuna

        assert val_set is not None
        Recommender.fit(self, train_set, val_set)

        compare_op = np.greater if self.metric.higher_better else np.less
        self.best_score = -np.inf if self.metric.higher_better else np.inf
        self.best_model = None
        best_number = None

        def objective(trial):
            nonlocal best_number
            params = {}
            for d in self.space:
                if isinstance(d, Discrete):
                    params[d.name] = trial.suggest_categorical(d.name, d.values)
                else:
                    params[d.name] = trial.suggest_float(
                        d.name, d.low, d.high, log=d.log
                    )
            if self.verbose:
                print("Evaluating: {}".format(params))

            model = self._fit_trial(trial, params, train_set, val_set)
            score = self._evaluate(model, train_set, val_set)
            if compare_op(score, self.best_score):
                self.best_score = score
                self.best_model = model
                best_number = trial.number
            return score

        sampler = self.sampler
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=getattr(self.model, "seed", None))

        verbosity = optuna.logging.get_verbosity()
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            self.study = optuna.create_study(
                storage=self.storage,
                sampler=sampler,
                pruner=self.pruner,
                study_name=self.study_name,
                direction="maximize" if self.metric.higher_better else "minimize",
                load_if_exists=True,
            )
            self.study.optimize(objective, n_trials=self.n_trials)
        finally:
            optuna.logging.set_verbosity(verbosity)

        best_trial = self.study.best_trial
        self.best_params = best_trial.params
        self.best_score = best_trial.value
        if best_trial.number != best_number:
            # the best trial comes from an earlier run of a resumed study
            self.best_model = self.model.clone(self.best_params).fit(train_set, val_set)

        if self.verbose:
            print("Best parameter settings: {}".format(self.best_params))
            print("{} = {:.4f}".format(self.metric.name, self.best_score))

        return self
