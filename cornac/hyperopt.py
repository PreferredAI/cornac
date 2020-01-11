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

from .models import Recommender
from .metrics import RatingMetric, RankingMetric
from .eval_methods import rating_eval, ranking_eval
from .utils import get_rng


__all__ = ["Discrete", "Continuous", "GridSearch", "RandomSearch"]


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
            
    """

    def __init__(self, name, low=0.0, high=1.0):
        super().__init__(name=name)
        self.low = low
        self.high = high

    def _sample(self, rng):
        """Sample a value of parameter used for RandomSearch"""
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

            if isinstance(self.metric, RatingMetric):
                score = rating_eval(model, [self.metric], val_set)[0][0]
            else:
                score = ranking_eval(
                    model,
                    [self.metric],
                    train_set,
                    val_set,
                    rating_threshold=self.eval_method.rating_threshold,
                    exclude_unknowns=self.eval_method.exclude_unknowns,
                    verbose=False,
                )[0][0]

            if compare_op(score, self.best_score):
                self.best_score = score
                self.best_model = model
                self.best_params = params

            del model

        if self.verbose:
            print("Best parameter settings: {}".format(self.best_params))
            print("{} = {:.4f}".format(self.metric.name, self.best_score))

        return self

    def score(self, user_idx, item_idx=None):
        """Scoring using the best searched model"""
        return self.best_model.score(user_idx, item_idx)


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
        rng = get_rng(self.model.seed)
        while len(param_set) < self.n_trails:
            params = [d._sample(rng) for d in self.space]
            param_set.append(dict(zip(keys, params)))
        return param_set
