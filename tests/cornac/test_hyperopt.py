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

import copy
import os
import tempfile
import unittest

import numpy as np
import numpy.testing as npt

from cornac.data import Reader
from cornac.models import MF, BPR, SPop, Recommender
from cornac.metrics import RMSE, AUC, HitRatio
from cornac.eval_methods import RatioSplit, NextItemEvaluation
from cornac.hyperopt import Discrete, Continuous
from cornac.hyperopt import GridSearch, RandomSearch, OptunaSearch
from cornac import Experiment
from cornac.utils import get_rng

try:
    import optuna
except ImportError:
    optuna = None


class ShiftedMean(Recommender):
    """Predict the global mean plus `shift`, checking for early stopping each epoch."""

    def __init__(self, shift=0.0, max_iter=3):
        super().__init__(name="ShiftedMean")
        self.shift = shift
        self.max_iter = max_iter

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)
        for _ in range(self.max_iter):
            if self.early_stop(train_set, val_set, patience=self.max_iter):
                break
        return self

    def monitor_value(self, train_set, val_set):
        return -abs(self.shift)

    def score(self, user_idx, item_idx=None):
        if item_idx is None:
            return np.full(self.total_items, self.global_mean + self.shift)
        return self.global_mean + self.shift


class TestCommon(unittest.TestCase):
    
    def setUp(self):
        data = Reader().read("./tests/data.txt")
        self.eval_method = RatioSplit(
            data, test_size=0.2, val_size=0.2, exclude_unknowns=False
        )

    def test_grid_search(self):
        model = MF(max_iter=1, verbose=True)
        metric = RMSE()
        gs_mf = GridSearch(
            model=model,
            space=[Discrete("k", [1, 2, 3]), Discrete("learning_rate", [0.1, 0.01])],
            metric=metric,
            eval_method=self.eval_method,
        )
        Experiment(
            eval_method=self.eval_method,
            models=[gs_mf],
            metrics=[metric],
            user_based=False,
        ).run()

    def test_random_search(self):
        model = BPR(max_iter=1, verbose=True)
        metric = AUC()
        rs_bpr = RandomSearch(
            model=model,
            space=[
                Discrete("k", [1, 2, 3]),
                Continuous("learning_rate", low=0.01, high=0.1),
            ],
            metric=metric,
            eval_method=self.eval_method,
        )
        Experiment(
            eval_method=self.eval_method,
            models=[rs_bpr],
            metrics=[metric],
            user_based=False,
        ).run()

    def test_random_search_next_item_recommender(self):
        data = Reader().read("./tests/sequence.txt", fmt="USIT", sep=" ")
        eval_method = NextItemEvaluation.from_splits(
            train_data=data[:35],
            val_data=data[35:50],
            test_data=data[50:],
            fmt="USIT",
            exclude_unknowns=False,
            mode="next",
        )
        metric = HitRatio(k=5)
        spop = SPop()
        spop.seed = 123  # for reproducible RandomSearch sampling
        rs_spop = RandomSearch(
            model=spop,
            space=[Discrete("use_session_popularity", [False, True])],
            metric=metric,
            eval_method=eval_method,
            n_trails=2,
        )

        test_result, _ = eval_method.evaluate(
            model=rs_spop,
            metrics=[metric],
            user_based=False,
            show_validation=False,
        )

        self.assertIsNotNone(rs_spop.best_model)
        self.assertEqual(rs_spop.best_params, {"use_session_popularity": False})
        self.assertAlmostEqual(rs_spop.best_score, 11 / 12)
        self.assertTrue(np.isfinite(test_result.metric_avg_results["HitRatio@5"]))

    def test_grid_search_next_item_recommender(self):
        data = Reader().read("./tests/sequence.txt", fmt="USIT", sep=" ")
        eval_method = NextItemEvaluation.from_splits(
            train_data=data[:35],
            val_data=data[35:50],
            test_data=data[50:],
            fmt="USIT",
            exclude_unknowns=False,
            mode="next",
        )
        metric = HitRatio(k=5)
        gs_spop = GridSearch(
            model=SPop(),
            space=[Discrete("use_session_popularity", [False, True])],
            metric=metric,
            eval_method=eval_method,
        )

        test_result, _ = eval_method.evaluate(
            model=gs_spop,
            metrics=[metric],
            user_based=False,
            show_validation=False,
        )

        self.assertIsNotNone(gs_spop.best_model)
        self.assertEqual(gs_spop.best_params, {"use_session_popularity": False})
        self.assertAlmostEqual(gs_spop.best_score, 11 / 12)
        self.assertTrue(np.isfinite(test_result.metric_avg_results["HitRatio@5"]))


@unittest.skipIf(optuna is None, "optuna is not installed")
class TestOptunaSearch(unittest.TestCase):
    def setUp(self):
        data = Reader().read("./tests/data.txt")
        self.eval_method = RatioSplit(
            data, test_size=0.2, val_size=0.2, exclude_unknowns=False, seed=123
        )

    def test_rating(self):
        model = MF(max_iter=1, verbose=True)
        metric = RMSE()
        os_mf = OptunaSearch(
            model=model,
            space=[Discrete("k", [1, 2, 3]), Discrete("learning_rate", [0.1, 0.01])],
            metric=metric,
            eval_method=self.eval_method,
            n_trials=4,
        )
        Experiment(
            eval_method=self.eval_method,
            models=[os_mf],
            metrics=[metric],
            user_based=False,
        ).run()

        self.assertEqual(len(os_mf.study.trials), 4)
        self.assertEqual(os_mf.best_score, os_mf.study.best_value)
        self.assertEqual(os_mf.best_params, os_mf.study.best_params)

    def test_ranking(self):
        model = BPR(max_iter=1, verbose=True)
        metric = AUC()
        os_bpr = OptunaSearch(
            model=model,
            space=[
                Discrete("k", [1, 2, 3]),
                Continuous("learning_rate", low=0.01, high=0.1, log=True),
            ],
            metric=metric,
            eval_method=self.eval_method,
            n_trials=5,
        )
        Experiment(
            eval_method=self.eval_method,
            models=[os_bpr],
            metrics=[metric],
            user_based=False,
        ).run()

        self.assertEqual(len(os_bpr.study.trials), 5)
        for trial in os_bpr.study.trials:
            self.assertTrue(0.01 <= trial.params["learning_rate"] <= 0.1)

    def test_next_item_recommender(self):
        data = Reader().read("./tests/sequence.txt", fmt="USIT", sep=" ")
        eval_method = NextItemEvaluation.from_splits(
            train_data=data[:35],
            val_data=data[35:50],
            test_data=data[50:],
            fmt="USIT",
            exclude_unknowns=False,
            mode="next",
        )
        metric = HitRatio(k=5)
        os_spop = OptunaSearch(
            model=SPop(),
            space=[Discrete("use_session_popularity", [False, True])],
            metric=metric,
            eval_method=eval_method,
            sampler=optuna.samplers.GridSampler(
                {"use_session_popularity": [False, True]}
            ),
        )

        test_result, _ = eval_method.evaluate(
            model=os_spop,
            metrics=[metric],
            user_based=False,
            show_validation=False,
        )

        self.assertIsNotNone(os_spop.best_model)
        self.assertEqual(len(os_spop.study.trials), 2)
        self.assertAlmostEqual(os_spop.best_score, 11 / 12)
        self.assertTrue(np.isfinite(test_result.metric_avg_results["HitRatio@5"]))

    def test_log_sampling(self):
        rng = get_rng(123)
        domain = Continuous("learning_rate", low=1e-4, high=1e-1, log=True)
        samples = np.array([domain._sample(rng) for _ in range(1000)])
        self.assertTrue(np.all(samples >= 1e-4))
        self.assertTrue(np.all(samples < 1e-1))
        # half of the log-uniform samples fall below the geometric mean
        self.assertAlmostEqual(np.mean(samples < 10**-2.5), 0.5, delta=0.05)

    def test_resume(self):
        train_set, val_set = self.eval_method.train_set, self.eval_method.val_set
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:

            callbacks = [lambda model, epoch, logs: None]

            def search(n_trials):
                model = MF(max_iter=1, seed=123)
                model.callbacks = callbacks
                return OptunaSearch(
                    model=model,
                    space=[Discrete("k", [1, 2, 3])],
                    metric=RMSE(),
                    eval_method=self.eval_method,
                    n_trials=n_trials,
                    storage="sqlite:///" + os.path.join(tmp_dir, "study.db"),
                    study_name="mf",
                )

            first = search(3).fit(train_set, val_set)
            self.assertEqual(len(first.study.trials), 3)

            second = search(2).fit(train_set, val_set)
            self.assertEqual(len(second.study.trials), 5)
            self.assertEqual(second.best_model.k, second.best_params["k"])
            self.assertTrue(np.isfinite(second.score(0, 0)))

            # without new trials, the best model is fitted from the stored study
            third = search(0).fit(train_set, val_set)
            self.assertEqual(len(third.study.trials), 5)
            self.assertEqual(third.best_params, second.best_params)
            self.assertEqual(third.best_score, second.best_score)
            self.assertEqual(third.score(0, 0), second.score(0, 0))
            self.assertIs(third.best_model.callbacks, callbacks)
            self.assertEqual(
                third.best_score, third._evaluate(third.best_model, train_set, val_set)
            )

    def test_no_completed_trial(self):
        os_mf = OptunaSearch(
            model=MF(max_iter=1),
            space=[Discrete("k", [1, 2, 3])],
            metric=RMSE(),
            eval_method=self.eval_method,
            n_trials=0,
        )
        with self.assertRaises(ValueError):
            os_mf.fit(self.eval_method.train_set, self.eval_method.val_set)

    def test_pruning(self):
        for metric in [RMSE(), AUC()]:
            os_sm = OptunaSearch(
                model=ShiftedMean(),
                space=[Continuous("shift", low=-1.0, high=1.0)],
                metric=metric,
                eval_method=self.eval_method,
                n_trials=20,
                sampler=optuna.samplers.RandomSampler(seed=123),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=1, n_warmup_steps=0
                ),
            )
            os_sm.fit(self.eval_method.train_set, self.eval_method.val_set)

            trials = os_sm.study.trials
            pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
            complete = [
                t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            self.assertEqual(len(trials), 20)
            self.assertGreater(len(pruned), 0)
            self.assertGreater(len(complete), 0)
            # trials with a worse monitored value than the best one are pruned
            best_shift = min(abs(t.params["shift"]) for t in complete)
            for t in pruned:
                self.assertGreater(abs(t.params["shift"]), best_shift)
            self.assertEqual(len(os_sm.best_model.callbacks), 0)

    def test_copy_and_save(self):
        os_mf = OptunaSearch(
            model=MF(max_iter=1),
            space=[Discrete("k", [1, 2, 3])],
            metric=RMSE(),
            eval_method=self.eval_method,
            n_trials=2,
        )
        os_mf.fit(self.eval_method.train_set, self.eval_method.val_set)

        copied = copy.deepcopy(os_mf)
        self.assertEqual(copied.best_params, os_mf.best_params)
        self.assertEqual(copied.score(0, 0), os_mf.score(0, 0))

        with tempfile.TemporaryDirectory() as tmp_dir:
            loaded = Recommender.load(os_mf.save(tmp_dir))
            self.assertEqual(loaded.best_params, os_mf.best_params)
            self.assertEqual(loaded.score(0, 0), os_mf.score(0, 0))
            self.assertEqual(len(loaded.study.trials), 2)


if __name__ == "__main__":
    unittest.main()
