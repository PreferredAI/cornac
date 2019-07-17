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

from cornac.eval_methods import BaseMethod
from cornac.data import TextModality, ImageModality
from cornac.data import TrainSet, Reader
from cornac.metrics import MAE, AUC
from cornac.models import MF


class TestBaseMethod(unittest.TestCase):

    def test_init(self):
        bm = BaseMethod(None, verbose=True)
        self.assertFalse(bm.exclude_unknowns)
        self.assertEqual(bm.rating_threshold, 1.0)

    def test_trainset_none(self):
        bm = BaseMethod(None, verbose=True)
        try:
            bm.evaluate(None, {}, False)
        except ValueError:
            assert True

    def test_testset_none(self):
        bm = BaseMethod(None, verbose=True)
        bm.train_set = TrainSet(None, None)
        try:
            bm.evaluate(None, {}, False)
        except ValueError:
            assert True

    def test_from_splits(self):
        data = Reader().read('./tests/data.txt')
        try:
            BaseMethod.from_splits(train_data=None, test_data=None)
        except ValueError:
            assert True

        try:
            BaseMethod.from_splits(train_data=data, test_data=None)
        except ValueError:
            assert True

        bm = BaseMethod.from_splits(train_data=data, test_data=data)
        self.assertEqual(bm.total_users, 10)
        self.assertEqual(bm.total_items, 10)

        bm = BaseMethod.from_splits(train_data=data, test_data=data,
                                    val_data=data, verbose=True)
        self.assertEqual(bm.total_users, 10)
        self.assertEqual(bm.total_items, 10)

    def test_with_modalities(self):
        bm = BaseMethod()

        self.assertIsNone(bm.user_text)
        self.assertIsNone(bm.item_text)
        self.assertIsNone(bm.user_image)
        self.assertIsNone(bm.item_image)
        self.assertIsNone(bm.user_graph)
        self.assertIsNone(bm.item_graph)

        bm.user_text = TextModality()
        bm.item_image = ImageModality()
        bm._build_modalities()

        try:
            bm.user_text = ImageModality()
        except ValueError:
            assert True

        try:
           bm.item_text = ImageModality()
        except ValueError:
           assert True

        try:
            bm.user_image = TextModality()
        except ValueError:
            assert True

        try:
           bm.item_image = TextModality()
        except ValueError:
           assert True

        try:
            bm.user_graph = TextModality()
        except ValueError:
            assert True

        try:
            bm.item_graph = ImageModality()
        except ValueError:
            assert True

    def test_organize_metrics(self):
        bm = BaseMethod()

        rating_metrics, ranking_metrics = bm._organize_metrics([MAE(), AUC()])
        self.assertEqual(len(rating_metrics), 1)  # MAE
        self.assertEqual(len(ranking_metrics), 1)  # AUC

        try:
            bm._organize_metrics(None)
        except ValueError:
            assert True

    def test_evaluate(self):
        data = Reader().read('./tests/data.txt')
        bm = BaseMethod.from_splits(train_data=data, test_data=data)
        model = MF(k=1, max_iter=0)
        result = bm.evaluate(model, metrics=[MAE()], user_based=False)
        result.__str__()

if __name__ == '__main__':
    unittest.main()
