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

import unittest

from cornac.data import Reader
from cornac.eval_methods import RatioSplit, CrossValidation
from cornac.models import PMF
from cornac.metrics import MAE, RMSE, Recall, FMeasure
from cornac.experiment.experiment import Experiment


class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.data = Reader().read('./tests/data.txt')

    def test_with_ratio_split(self):
        exp = Experiment(eval_method=RatioSplit(self.data, verbose=True),
                         models=[PMF(1, 0)],
                         metrics=[MAE(), RMSE(), Recall(1), FMeasure(1)],
                         verbose=True)
        exp.run()

        try:
            Experiment(None, None, None)
        except ValueError:
            assert True

        try:
            Experiment(None, [PMF(1, 0)], None)
        except ValueError:
            assert True

    def test_with_cross_validation(self):
        exp = Experiment(eval_method=CrossValidation(self.data),
                         models=[PMF(1, 0)],
                         metrics=[MAE(), RMSE(), Recall(1), FMeasure(1)],
                         verbose=True)
        exp.run()


if __name__ == '__main__':
    unittest.main()
