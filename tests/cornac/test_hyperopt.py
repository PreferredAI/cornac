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

import numpy as np
import numpy.testing as npt

from cornac.data import Reader
from cornac.models import MF, BPR
from cornac.metrics import RMSE, AUC
from cornac.eval_methods import RatioSplit
from cornac.hyperopt import Discrete, Continuous
from cornac.hyperopt import GridSearch, RandomSearch
from cornac import Experiment


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


if __name__ == "__main__":
    unittest.main()
