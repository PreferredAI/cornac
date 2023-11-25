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

from cornac.metrics.ranking import RankingMetric
from cornac.metrics.ranking import MeasureAtK
from cornac.metrics import NDCG
from cornac.metrics import NCRR
from cornac.metrics import MRR
from cornac.metrics import HR
from cornac.metrics import Precision
from cornac.metrics import Recall
from cornac.metrics import FMeasure
from cornac.metrics import AUC
from cornac.metrics import MAP


class TestRanking(unittest.TestCase):
    def test_ranking_metric(self):
        metric = RankingMetric()

        self.assertEqual(metric.type, "ranking")
        self.assertIsNone(metric.name)
        self.assertEqual(metric.k, -1)

        try:
            metric.compute()
        except NotImplementedError:
            assert True

    def test_ndcg(self):
        ndcg = NDCG()

        self.assertEqual(ndcg.type, "ranking")
        self.assertEqual(ndcg.name, "NDCG@-1")

        self.assertEqual(
            1,
            ndcg.compute(gt_pos=np.asarray([0]), pd_rank=np.asarray([0])),
        )

        gt_pos = np.asarray([0, 2])  # [1, 3]
        pd_rank = np.asarray([0, 2, 1])  # [1, 3, 2]
        self.assertEqual(1, ndcg.compute(gt_pos, pd_rank))

        ndcg_2 = NDCG(k=2)
        self.assertEqual(ndcg_2.k, 2)

        gt_pos = np.asarray([2])  # [3]
        pd_rank = np.asarray([1, 2, 0])  # [2, 3, 1]
        self.assertEqual(
            0.63,
            float("{:.2f}".format(ndcg_2.compute(gt_pos, pd_rank))),
        )

    def test_ncrr(self):
        ncrr = NCRR()

        self.assertEqual(ncrr.type, "ranking")
        self.assertEqual(ncrr.name, "NCRR@-1")

        self.assertEqual(1, ncrr.compute(np.asarray([0]), np.asarray([0])))

        gt_pos = np.asarray([0, 2])  # [1, 3]
        pd_rank = np.asarray([0, 2, 1])  # [1, 3, 2]
        self.assertEqual(1, ncrr.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([0, 2])  # [1, 3]
        pd_rank = np.asarray([1, 2, 0])  # [2, 3, 1]
        self.assertEqual(((1 / 3 + 1 / 2) / (1 + 1 / 2)), ncrr.compute(gt_pos, pd_rank))

        ncrr_2 = NCRR(k=2)
        self.assertEqual(ncrr_2.k, 2)

        gt_pos = np.asarray([2])  # [3]
        pd_rank = np.asarray([1, 2, 0])  # [2, 3, 1]
        self.assertEqual(0.5, ncrr_2.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([2])  # [3]
        pd_rank = np.asarray([4, 1, 2])  # [5, 2, 3]
        self.assertEqual(0.0, ncrr_2.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([0, 1, 2])  # [1, 2, 3]
        pd_rank = np.asarray([5, 1, 6])  # [6, 2, 7]
        self.assertEqual(1.0 / 3.0, ncrr_2.compute(gt_pos, pd_rank))

        ncrr_3 = NCRR(k=3)
        gt_pos = np.asarray([0, 1])  # [1, 2]
        pd_rank = np.asarray([5, 1, 6, 8])  # [6, 2, 7, 9]
        self.assertEqual(1.0 / 3.0, ncrr_3.compute(gt_pos, pd_rank))

    def test_mrr(self):
        mrr = MRR()

        self.assertEqual(mrr.type, "ranking")
        self.assertEqual(mrr.name, "MRR")

        self.assertEqual(1, mrr.compute(np.asarray([0]), np.asarray([0])))

        gt_pos = np.asarray([0, 2])  # [1, 3]
        pd_rank = np.asarray([0, 2, 1])  # [1, 3, 2]
        self.assertEqual(1, mrr.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([0, 2])  # [1, 3]
        pd_rank = np.asarray([1, 2, 0])  # [2, 3, 1]
        self.assertEqual(1 / 2, mrr.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([0, 2])  # [1, 3]
        pd_rank = np.asarray([1])  # [2]
        try:
            mrr.compute(gt_pos, pd_rank)
        except ValueError:
            assert True

    def test_measure_at_k(self):
        measure_at_k = MeasureAtK()

        self.assertEqual(measure_at_k.type, "ranking")
        assert measure_at_k.name is None
        self.assertEqual(measure_at_k.k, -1)

        tp, tp_fn, tp_fp = measure_at_k.compute(np.asarray([0]), np.asarray([0]))
        self.assertEqual(1, tp)
        self.assertEqual(1, tp_fn)
        self.assertEqual(1, tp_fp)

        gt_pos = np.asarray([0, 2])  # [1, 0, 1]
        pd_rank = np.asarray([0, 2, 1])  # [1, 1, 1]
        tp, tp_fn, tp_fp = measure_at_k.compute(gt_pos, pd_rank)
        self.assertEqual(2, tp)
        self.assertEqual(2, tp_fn)
        self.assertEqual(3, tp_fp)

    def test_hit_ratio(self):
        hr = HR()

        self.assertEqual(hr.type, "ranking")
        self.assertEqual(hr.name, "HR@-1")

        self.assertEqual(1, hr.compute(np.asarray([0]), np.asarray([0])))
        self.assertEqual(1, hr.compute(np.asarray([0, 1]), np.asarray([0, 2])))

        gt_pos = np.asarray([0, 2])  # [1, 0, 1]
        pd_rank = np.asarray([0, 2, 1])  # [1, 1, 1]
        self.assertEqual(1, hr.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([2])  # [0, 0, 1]
        pd_rank = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual(1, hr.compute(gt_pos, pd_rank))

        hr_2 = HR(k=2)
        self.assertEqual(hr_2.k, 2)

        gt_pos = np.asarray([0])  # [0, 0, 1]
        pd_rank = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual(0, hr_2.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([2])  # [0, 0, 1]
        pd_rank = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual(1, hr_2.compute(gt_pos, pd_rank))

    def test_precision(self):
        prec = Precision()

        self.assertEqual(prec.type, "ranking")
        self.assertEqual(prec.name, "Precision@-1")

        self.assertEqual(1, prec.compute(np.asarray([0]), np.asarray([0])))

        gt_pos = np.asarray([0, 2])  # [1, 0, 1]
        pd_rank = np.asarray([0, 2, 1])  # [1, 1, 1]
        self.assertEqual((2 / 3), prec.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([2])  # [0, 0, 1]
        pd_rank = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual((1 / 3), prec.compute(gt_pos, pd_rank))

        prec_2 = Precision(k=2)
        self.assertEqual(prec_2.k, 2)

        gt_pos = np.asarray([2])  # [0, 0, 1]
        pd_rank = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual(0.5, prec_2.compute(gt_pos, pd_rank))

    def test_recall(self):
        rec = Recall()

        self.assertEqual(rec.type, "ranking")
        self.assertEqual(rec.name, "Recall@-1")

        self.assertEqual(1, rec.compute(np.asarray([0]), np.asarray([0])))

        gt_pos = np.asarray([0, 2])  # [1, 0, 1]
        pd_rank = np.asarray([0, 2, 1])  # [1, 1, 1]
        self.assertEqual(1, rec.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([2])  # [0, 0, 1]
        pd_rank = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual(1, rec.compute(gt_pos, pd_rank))

        rec_2 = Recall(k=2)
        self.assertEqual(rec_2.k, 2)

        gt_pos = np.asarray([2])  # [0, 0, 1]
        pd_rank = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual(1, rec_2.compute(gt_pos, pd_rank))

    def test_f_measure(self):
        f1 = FMeasure()

        self.assertEqual(f1.type, "ranking")
        self.assertEqual(f1.name, "F1@-1")

        self.assertEqual(1, f1.compute(np.asarray([0]), np.asarray([0])))

        gt_pos = np.asarray([0, 2])  # [1, 0, 1]
        pd_rank = np.asarray([0, 2, 1])  # [1, 1, 1]
        self.assertEqual((4 / 5), f1.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([2])  # [0, 0, 1]
        pd_rank = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual((1 / 2), f1.compute(gt_pos, pd_rank))

        f1_2 = FMeasure(k=2)
        self.assertEqual(f1_2.k, 2)

        gt_pos = np.asarray([2])  # [0, 0, 1]
        pd_rank = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual((2 / 3), f1_2.compute(gt_pos, pd_rank))

        gt_pos = np.asarray([0])  # [1, 0, 0]
        pd_rank = np.asarray([1, 2])  # [0, 1, 1]
        self.assertEqual(0, f1_2.compute(gt_pos, pd_rank))

    def test_auc(self):
        auc = AUC()

        self.assertEqual(auc.type, "ranking")
        self.assertEqual(auc.name, "AUC")

        item_indices = np.arange(4)
        gt_pos = np.array([2, 3])  # [0, 0, 1, 1]
        pd_scores = np.array([0.1, 0.4, 0.35, 0.8])
        auc_score = auc.compute(item_indices, pd_scores, gt_pos)
        self.assertEqual(0.75, auc_score)

        item_indices = np.arange(4)
        gt_pos = np.array([1, 3])  # [0, 1, 0, 1]
        pd_scores = np.array([0.1, 0.4, 0.35, 0.8])
        auc_score = auc.compute(item_indices, pd_scores, gt_pos)
        self.assertEqual(1.0, auc_score)

        gt_pos = np.array([2])  # [0, 0, 1, 0]
        gt_neg = np.array([1, 1, 0, 0])
        pd_scores = np.array([0.1, 0.4, 0.35, 0.8])
        auc_score = auc.compute(item_indices, pd_scores, gt_pos, gt_neg)
        self.assertEqual(0.5, auc_score)

    def test_map(self):
        mAP = MAP()

        self.assertEqual(mAP.type, "ranking")
        self.assertEqual(mAP.name, "MAP")

        item_indices = np.arange(3)
        gt_pos = np.array([0])  # [1, 0, 0]
        pd_scores = np.array([0.75, 0.5, 1])
        self.assertEqual(0.5, mAP.compute(item_indices, pd_scores, gt_pos))

        item_indices = np.arange(3)
        gt_pos = np.array([2])  # [0, 0, 1]
        pd_scores = np.array([1, 0.2, 0.1])
        self.assertEqual(1 / 3, mAP.compute(item_indices, pd_scores, gt_pos))

        item_indices = np.arange(10)
        gt_pos = np.array([1, 3, 5])  # [0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
        pd_scores = np.linspace(0.0, 1.0, len(item_indices))[::-1]
        self.assertEqual(0.5, mAP.compute(item_indices, pd_scores, gt_pos))


if __name__ == "__main__":
    unittest.main()
