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

import cornac
from cornac.data import Reader
from cornac.eval_methods import PropensityStratifiedEvaluation
from cornac.models import MF
from cornac.metrics import MAE, AUC


class TestPropensityStratifiedEvaluation(unittest.TestCase):
    def setUp(self):
        self.ml_100k = cornac.datasets.movielens.load_feedback()
        cutoff = int(len(self.ml_100k) * 0.1)  # use 10% for faster testing
        self.ml_100k = self.ml_100k[:cutoff]

    def test_stratified_split(self, n_strata=2):
        stra_eval_method = PropensityStratifiedEvaluation(
            data=self.ml_100k, n_strata=n_strata, rating_threshold=4.0, verbose=True
        )
        strata = [f"Q{idx+1}" for idx in range(n_strata)]
        # total number of ratings in the test set should be split
        # within different strata
        num_ratings = 0
        for stratum in strata:
            if stratum in stra_eval_method.stratified_sets.keys():
                num_ratings += stra_eval_method.stratified_sets[stratum].num_ratings
        self.assertEqual(num_ratings, stra_eval_method.test_set.num_ratings)

        # the number of sampled user/items in each stratum should be lower than
        # the total number of users/items in the test set
        total_users = len(stra_eval_method.test_set.uid_map)
        total_items = len(stra_eval_method.test_set.iid_map)
        for stratum in strata:
            if stratum in stra_eval_method.stratified_sets.keys():
                strata_num_users = len(
                    stra_eval_method.stratified_sets[stratum].uid_map
                )
                self.assertTrue(strata_num_users <= total_users)
                strata_num_items = len(
                    stra_eval_method.stratified_sets[stratum].iid_map
                )
                self.assertTrue(strata_num_items <= total_items)

    def test_propensity(self, n_strata=2):
        stra_eval_method = PropensityStratifiedEvaluation(
            data=self.ml_100k, n_strata=n_strata, rating_threshold=4.0, verbose=True
        )
        props = np.array(list(stra_eval_method.props.values()))
        self.assertTrue(np.all(props > 0))

    def test_strata(self):
        for n_strata in range(2, 5):
            self.test_propensity(n_strata)
            self.test_stratified_split(n_strata)

    def test_evaluate(self, n_strata=2):
        stra_eval_method = PropensityStratifiedEvaluation(
            data=self.ml_100k, val_size=0.1, n_strata=n_strata, rating_threshold=4.0, verbose=True
        )
        model = MF(k=1, max_iter=0)
        result = stra_eval_method.evaluate(
            model, metrics=[MAE(), AUC()], user_based=False
        )
        result.__str__()


if __name__ == "__main__":
    unittest.main()
