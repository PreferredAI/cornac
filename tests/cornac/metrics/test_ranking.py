# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest
import numpy as np
from cornac.metrics.ranking import RankingMetric
from cornac.metrics.ranking import MeasureAtK
from cornac.metrics import NDCG
from cornac.metrics import NCRR
from cornac.metrics import MRR
from cornac.metrics import Precision
from cornac.metrics import Recall
from cornac.metrics import FMeasure
from cornac.metrics import AUC


class TestRanking(unittest.TestCase):

    def test_ranking_metric(self):
        metric = RankingMetric()

        self.assertEqual(metric.type, 'ranking')
        self.assertIsNone(metric.name)
        self.assertEqual(metric.k, -1)

        try:
            metric.compute()
        except NotImplementedError:
            assert True

    def test_ndcg(self):
        ndcg = NDCG()

        self.assertEqual(ndcg.type, 'ranking')
        self.assertEqual(ndcg.name, 'NDCG@-1')

        self.assertEqual(1, ndcg.compute(np.asarray([1]), np.asarray([0])))

        ground_truth = np.asarray([1, 0, 1])  # [1, 3]
        rec_list = np.asarray([0, 2, 1])  # [1, 3, 2]
        self.assertEqual(1, ndcg.compute(ground_truth, rec_list))

        ndcg_2 = NDCG(k=2)
        self.assertEqual(ndcg_2.k, 2)

        ground_truth = np.asarray([0, 0, 1])  # [3]
        rec_list = np.asarray([1, 2, 0])  # [2, 3, 1]
        self.assertEqual(0.63, float('{:.2f}'.format(ndcg_2.compute(ground_truth, rec_list))))

    def test_ncrr(self):
        ncrr = NCRR()

        self.assertEqual(ncrr.type, 'ranking')
        self.assertEqual(ncrr.name, 'NCRR@-1')

        self.assertEqual(1, ncrr.compute(np.asarray([1]), np.asarray([0])))

        ground_truth = np.asarray([1, 0, 1])  # [1, 3]
        rec_list = np.asarray([0, 2, 1])  # [1, 3, 2]
        self.assertEqual(1, ncrr.compute(ground_truth, rec_list))

        ground_truth = np.asarray([1, 0, 1])  # [1, 3]
        rec_list = np.asarray([1, 2, 0])  # [2, 3, 1]
        self.assertEqual(((1 / 3 + 1 / 2) / (1 + 1 / 2)), ncrr.compute(ground_truth, rec_list))

        ncrr_2 = NCRR(k=2)
        self.assertEqual(ncrr_2.k, 2)

        ground_truth = np.asarray([0, 0, 1])  # [3]
        rec_list = np.asarray([1, 2, 0])  # [2, 3, 1]
        self.assertEqual(0.5, ncrr_2.compute(ground_truth, rec_list))

    def test_mrr(self):
        mrr = MRR()

        self.assertEqual(mrr.type, 'ranking')
        self.assertEqual(mrr.name, 'MRR')

        self.assertEqual(1, mrr.compute(np.asarray([1]), np.asarray([0])))

        ground_truth = np.asarray([1, 0, 1])  # [1, 3]
        rec_list = np.asarray([0, 2, 1])  # [1, 3, 2]
        self.assertEqual(1, mrr.compute(ground_truth, rec_list))

        ground_truth = np.asarray([1, 0, 1])  # [1, 3]
        rec_list = np.asarray([1, 2, 0])  # [2, 3, 1]
        self.assertEqual(1 / 2, mrr.compute(ground_truth, rec_list))

        ground_truth = np.asarray([1, 0, 1])  # [1, 3]
        rec_list = np.asarray([1])  # [2]
        try:
            mrr.compute(ground_truth, rec_list)
        except ValueError:
            assert True

    def test_measure_at_k(self):
        measure_at_k = MeasureAtK()

        self.assertEqual(measure_at_k.type, 'ranking')
        assert measure_at_k.name is None
        self.assertEqual(measure_at_k.k, -1)

        tp, tp_fn, tp_fp = measure_at_k.compute(np.asarray([1]), np.asarray([0]))
        self.assertEqual(1, tp)
        self.assertEqual(1, tp_fn)
        self.assertEqual(1, tp_fp)

        ground_truth = np.asarray([1, 0, 1])  # [1, 0, 1]
        rec_list = np.asarray([0, 2, 1])  # [1, 1, 1]
        tp, tp_fn, tp_fp = measure_at_k.compute(ground_truth, rec_list)
        self.assertEqual(2, tp)
        self.assertEqual(2, tp_fn)
        self.assertEqual(3, tp_fp)

    def test_precision(self):
        prec = Precision()

        self.assertEqual(prec.type, 'ranking')
        self.assertEqual(prec.name, 'Precision@-1')

        self.assertEqual(1, prec.compute(np.asarray([1]), np.asarray([0])))

        ground_truth = np.asarray([1, 0, 1])  # [1, 0, 1]
        rec_list = np.asarray([0, 2, 1])  # [1, 1, 1]
        self.assertEqual((2 / 3), prec.compute(ground_truth, rec_list))

        ground_truth = np.asarray([0, 0, 1])  # [0, 0, 1]
        rec_list = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual((1 / 3), prec.compute(ground_truth, rec_list))

        prec_2 = Precision(k=2)
        self.assertEqual(prec_2.k, 2)

        ground_truth = np.asarray([0, 0, 1])  # [0, 0, 1]
        rec_list = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual(0.5, prec_2.compute(ground_truth, rec_list))

    def test_recall(self):
        rec = Recall()

        self.assertEqual(rec.type, 'ranking')
        self.assertEqual(rec.name, 'Recall@-1')

        self.assertEqual(1, rec.compute(np.asarray([1]), np.asarray([0])))

        ground_truth = np.asarray([1, 0, 1])  # [1, 0, 1]
        rec_list = np.asarray([0, 2, 1])  # [1, 1, 1]
        self.assertEqual(1, rec.compute(ground_truth, rec_list))

        ground_truth = np.asarray([0, 0, 1])  # [0, 0, 1]
        rec_list = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual(1, rec.compute(ground_truth, rec_list))

        rec_2 = Recall(k=2)
        self.assertEqual(rec_2.k, 2)

        ground_truth = np.asarray([0, 0, 1])  # [0, 0, 1]
        rec_list = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual(1, rec_2.compute(ground_truth, rec_list))

    def test_f_measure(self):
        f1 = FMeasure()

        self.assertEqual(f1.type, 'ranking')
        self.assertEqual(f1.name, 'F1@-1')

        self.assertEqual(1, f1.compute(np.asarray([1]), np.asarray([0])))

        ground_truth = np.asarray([1, 0, 1])  # [1, 0, 1]
        rec_list = np.asarray([0, 2, 1])  # [1, 1, 1]
        self.assertEqual((4 / 5), f1.compute(ground_truth, rec_list))

        ground_truth = np.asarray([0, 0, 1])  # [0, 0, 1]
        rec_list = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual((1 / 2), f1.compute(ground_truth, rec_list))

        f1_2 = FMeasure(k=2)
        self.assertEqual(f1_2.k, 2)

        ground_truth = np.asarray([0, 0, 1])  # [0, 0, 1]
        rec_list = np.asarray([1, 2, 0])  # [1, 1, 1]
        self.assertEqual((2 / 3), f1_2.compute(ground_truth, rec_list))

        ground_truth = np.asarray([1, 0, 0])  # [1, 0, 0]
        rec_list = np.asarray([1, 2])  # [0, 1, 1]
        self.assertEqual(0, f1_2.compute(ground_truth, rec_list))

    def test_auc(self):
        auc = AUC()

        self.assertEqual(auc.type, 'ranking')
        self.assertEqual(auc.name, 'AUC')

        gt_pos = np.array([0, 0, 1, 1])
        pd_scores = np.array([0.1, 0.4, 0.35, 0.8])
        auc_score = auc.compute(pd_scores, gt_pos)
        self.assertEqual(0.75, auc_score)

        gt_pos = np.array([0, 1, 0, 1])
        pd_scores = np.array([0.1, 0.4, 0.35, 0.8])
        auc_score = auc.compute(pd_scores, gt_pos)
        self.assertEqual(1.0, auc_score)

        gt_pos = np.array([0, 0, 1, 0])
        gt_neg = np.array([1, 1, 0, 0])
        pd_scores = np.array([0.1, 0.4, 0.35, 0.8])
        auc_score = auc.compute(pd_scores, gt_pos, gt_neg)
        self.assertEqual(0.5, auc_score)


if __name__ == '__main__':
    unittest.main()
