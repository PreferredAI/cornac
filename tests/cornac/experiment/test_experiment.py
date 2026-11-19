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

import json
import os
import tempfile
import unittest
from collections import OrderedDict

import numpy as np

from cornac.data import Reader
from cornac.eval_methods import RatioSplit, CrossValidation
from cornac.models import PMF
from cornac.metrics import MAE, RMSE, Recall, FMeasure
from cornac.experiment.experiment import Experiment
from cornac.experiment.result import (
    Result,
    PSTResult,
    ExperimentResult,
    CVExperimentResult,
)

try:
    import pandas
except ImportError:
    pandas = None


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.data = Reader().read("./tests/data.txt")

    def test_with_ratio_split(self):
        Experiment(
            eval_method=RatioSplit(
                self.data + [(self.data[0][0], self.data[1][1], 5.0)],
                exclude_unknowns=True,
                seed=123,
                verbose=True,
            ),
            models=[PMF(1, 0)],
            metrics=[MAE(), RMSE()],
            verbose=True,
        ).run()

        try:
            Experiment(None, None, None)
        except ValueError:
            assert True

        try:
            Experiment(None, [PMF(1, 0)], None)
        except ValueError:
            assert True

    def test_with_cross_validation(self):
        Experiment(
            eval_method=CrossValidation(
                self.data + [(self.data[0][0], self.data[1][1], 5.0)],
                exclude_unknowns=False,
                verbose=True,
            ),
            models=[PMF(1, 0)],
            metrics=[Recall(1), FMeasure(1)],
            verbose=True,
        ).run()

    def test_result_export(self):
        ratio_split = RatioSplit(
            self.data, test_size=0.2, val_size=0.2, exclude_unknowns=False, seed=123
        )
        fitted = PMF(k=2, max_iter=1, seed=123).fit(ratio_split.train_set)
        exp = Experiment(
            eval_method=ratio_split,
            models=[PMF(k=2, max_iter=1, init_params={"U": fitted.U, "V": fitted.V})],
            metrics=[MAE(), RMSE()],
        )
        exp.run()

        for result in exp.result + exp.val_result:
            self.assertEqual(result.model_params["k"], 2)
            self.assertNotIn("init_params", result.model_params)
            self.assertEqual(result.eval_params, {"method": "RatioSplit", "seed": 123})

        expected = [r.to_dict() for r in exp.result]
        self.assertEqual(
            list(expected[0]["metrics"]), ["MAE", "RMSE", "Train (s)", "Test (s)"]
        )
        self.assertEqual(json.loads(exp.result.to_json()), expected)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "result.json")
            exp.result.to_json(path)
            with open(path) as f:
                self.assertEqual(json.load(f), expected)

    @unittest.skipIf(pandas is None, "pandas is not installed")
    def test_result_to_dataframe(self):
        exp = Experiment(
            eval_method=RatioSplit(
                self.data, test_size=0.2, val_size=0.2, exclude_unknowns=False, seed=123
            ),
            models=[
                PMF(k=1, max_iter=0, name="PMF1"),
                PMF(k=2, max_iter=0, name="PMF2"),
            ],
            metrics=[MAE(), RMSE()],
        )
        exp.run()

        for results in (exp.result, exp.val_result):
            df = results.to_dataframe()
            self.assertEqual(df.index.name, "model")
            self.assertEqual(list(df.index), ["PMF1", "PMF2"])
            for r in results:
                self.assertEqual(list(df.columns), list(r.metric_avg_results))
                self.assertEqual(df.loc[r.model_name].to_dict(), r.metric_avg_results)

        results = ExperimentResult(
            [
                Result("A", OrderedDict([("MAE", 0.5)]), None),
                Result("B", OrderedDict([("RMSE", 0.7), ("MAE", 0.6)]), None),
            ]
        )
        df = results.to_dataframe()
        self.assertEqual(list(df.columns), ["MAE", "RMSE"])
        self.assertTrue(np.isnan(df.loc["A", "RMSE"]))

    @unittest.skipIf(pandas is None, "pandas is not installed")
    def test_cv_result_export(self):
        exp = Experiment(
            eval_method=CrossValidation(
                self.data + [(self.data[0][0], self.data[1][1], 5.0)],
                exclude_unknowns=False,
                seed=123,
            ),
            models=[PMF(1, 0)],
            metrics=[Recall(1), FMeasure(1)],
        )
        exp.run()

        result = exp.result[0]
        self.assertEqual(result.model_params["k"], 1)
        self.assertEqual(result.eval_params, {"method": "CrossValidation", "seed": 123})
        rows = ["Fold %d" % f for f in range(5)] + ["Mean", "Std"]
        self.assertEqual(list(result.to_dict()["metrics"]), rows)
        self.assertEqual(json.loads(exp.result.to_json()), [result.to_dict()])

        df = exp.result.to_dataframe()
        self.assertEqual(list(df.index), [("PMF", row) for row in rows])
        self.assertEqual(list(df.index.names), ["model", None])
        self.assertEqual(df.loc[("PMF", "Mean")].to_dict(), result.metric_mean)

    @unittest.skipIf(pandas is None, "pandas is not installed")
    def test_pst_result_export(self):
        result = PSTResult("PMF")
        for mae, size in [(0.5, 4), (0.4, 4), (0.3, 3), (0.2, 1)]:
            avg = OrderedDict([("MAE", mae), ("SIZE", size)])
            result.append(Result("PMF", avg, None))
        result.organize()

        metrics = result.to_dict()["metrics"]
        self.assertEqual(list(metrics), ["Closed", "IPS", "Q1", "Q2", "Unbiased"])
        self.assertAlmostEqual(metrics["Unbiased"]["MAE"], 0.275)
        self.assertEqual(metrics["Unbiased"]["SIZE"], 4.0)

        df = CVExperimentResult([result]).to_dataframe()
        self.assertEqual(df.shape, (5, 2))
        self.assertEqual(df.loc[("PMF", "Q1"), "MAE"], 0.3)


if __name__ == "__main__":
    unittest.main()
