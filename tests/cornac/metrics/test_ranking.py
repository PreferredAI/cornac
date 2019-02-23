# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np

from cornac.metrics.ranking import RankingMetric
from cornac.metrics.ranking import MeasureAtK
from cornac.metrics import NDCG
from cornac.metrics import NCRR
from cornac.metrics import MRR
from cornac.metrics import Precision
from cornac.metrics import Recall
from cornac.metrics import FMeasure

def test_ranking_metric():
    metric = RankingMetric()

    assert metric.type == 'ranking'
    assert metric.name is None
    assert metric.k == -1

    try:
        metric.compute()
    except NotImplementedError:
        assert True


def test_ndcg():
    ndcg = NDCG()

    assert ndcg.type == 'ranking'
    assert ndcg.name == 'NDCG@-1'

    assert 1 == ndcg.compute(np.asarray([1]), np.asarray([0]))

    ground_truth = np.asarray([1, 0, 1]) # [1, 3]
    rec_list = np.asarray([0, 2, 1]) # [1, 3, 2]
    assert 1 == ndcg.compute(ground_truth, rec_list)

    ndcg_2 = NDCG(k=2)
    assert ndcg_2.k == 2

    ground_truth = np.asarray([0, 0, 1]) # [3]
    rec_list = np.asarray([1, 2, 0]) # [2, 3, 1]
    assert 0.63 == float('{:.2f}'.format(ndcg_2.compute(ground_truth, rec_list)))


def test_ncrr():
    ncrr = NCRR()

    assert ncrr.type == 'ranking'
    assert ncrr.name == 'NCRR@-1'

    assert 1 == ncrr.compute(np.asarray([1]), np.asarray([0]))

    ground_truth = np.asarray([1, 0, 1]) # [1, 3]
    rec_list = np.asarray([0, 2, 1]) # [1, 3, 2]
    assert 1 == ncrr.compute(ground_truth, rec_list)

    ground_truth = np.asarray([1, 0, 1]) # [1, 3]
    rec_list = np.asarray([1, 2, 0]) # [2, 3, 1]
    assert ((1/3 + 1/2) / (1 + 1/2)) == ncrr.compute(ground_truth, rec_list)

    ncrr_2 = NCRR(k=2)
    assert ncrr_2.k == 2

    ground_truth = np.asarray([0, 0, 1]) # [3]
    rec_list = np.asarray([1, 2, 0]) # [2, 3, 1]
    assert 0.5 == ncrr_2.compute(ground_truth, rec_list)


def test_mrr():
    mrr = MRR()

    assert mrr.type == 'ranking'
    assert mrr.name == 'MRR'

    assert 1 == mrr.compute(np.asarray([1]), np.asarray([0]))

    ground_truth = np.asarray([1, 0, 1]) # [1, 3]
    rec_list = np.asarray([0, 2, 1]) # [1, 3, 2]
    assert 1 == mrr.compute(ground_truth, rec_list)

    ground_truth = np.asarray([1, 0, 1]) # [1, 3]
    rec_list = np.asarray([1, 2, 0]) # [2, 3, 1]
    assert 1/2 == mrr.compute(ground_truth, rec_list)

    ground_truth = np.asarray([1, 0, 1]) # [1, 3]
    rec_list = np.asarray([1]) # [2]
    try:
        mrr.compute(ground_truth, rec_list)
    except ValueError:
        assert True


def test_measure_at_k():
    measure_at_k = MeasureAtK()

    assert measure_at_k.type == 'ranking'
    assert measure_at_k.name is None
    assert measure_at_k.k == -1

    tp, tp_fn, tp_fp = measure_at_k.compute(np.asarray([1]), np.asarray([0]))
    assert 1 == tp
    assert 1 == tp_fn
    assert 1 == tp_fp

    ground_truth = np.asarray([1, 0, 1]) # [1, 0, 1]
    rec_list = np.asarray([0, 2, 1]) # [1, 1, 1]
    tp, tp_fn, tp_fp = measure_at_k.compute(ground_truth, rec_list)
    assert 2 == tp
    assert 2 == tp_fn
    assert 3 == tp_fp


def test_precision():
    prec = Precision()

    assert prec.type == 'ranking'
    assert prec.name == 'Precision@-1'

    assert 1 == prec.compute(np.asarray([1]), np.asarray([0]))

    ground_truth = np.asarray([1, 0, 1]) # [1, 0, 1]
    rec_list = np.asarray([0, 2, 1]) # [1, 1, 1]
    assert (2/3) == prec.compute(ground_truth, rec_list)

    ground_truth = np.asarray([0, 0, 1]) # [0, 0, 1]
    rec_list = np.asarray([1, 2, 0]) # [1, 1, 1]
    assert (1/3) == prec.compute(ground_truth, rec_list)

    prec_2 = Precision(k=2)
    assert prec_2.k == 2

    ground_truth = np.asarray([0, 0, 1]) # [0, 0, 1]
    rec_list = np.asarray([1, 2, 0]) # [1, 1, 1]
    assert 0.5 == prec_2.compute(ground_truth, rec_list)


def test_recall():
    rec = Recall()

    assert rec.type == 'ranking'
    assert rec.name == 'Recall@-1'

    assert 1 == rec.compute(np.asarray([1]), np.asarray([0]))

    ground_truth = np.asarray([1, 0, 1]) # [1, 0, 1]
    rec_list = np.asarray([0, 2, 1]) # [1, 1, 1]
    assert 1 == rec.compute(ground_truth, rec_list)

    ground_truth = np.asarray([0, 0, 1]) # [0, 0, 1]
    rec_list = np.asarray([1, 2, 0]) # [1, 1, 1]
    assert 1 == rec.compute(ground_truth, rec_list)

    rec_2 = Recall(k=2)
    assert rec_2.k == 2

    ground_truth = np.asarray([0, 0, 1]) # [0, 0, 1]
    rec_list = np.asarray([1, 2, 0]) # [1, 1, 1]
    assert 1 == rec_2.compute(ground_truth, rec_list)


def test_f_measure():
    f1 = FMeasure()

    assert f1.type == 'ranking'
    assert f1.name == 'F1@-1'

    assert 1 == f1.compute(np.asarray([1]), np.asarray([0]))

    ground_truth = np.asarray([1, 0, 1]) # [1, 0, 1]
    rec_list = np.asarray([0, 2, 1]) # [1, 1, 1]
    assert (4/5) == f1.compute(ground_truth, rec_list)

    ground_truth = np.asarray([0, 0, 1]) # [0, 0, 1]
    rec_list = np.asarray([1, 2, 0]) # [1, 1, 1]
    assert (1/2) == f1.compute(ground_truth, rec_list)

    f1_2 = FMeasure(k=2)
    assert f1_2.k == 2

    ground_truth = np.asarray([0, 0, 1]) # [0, 0, 1]
    rec_list = np.asarray([1, 2, 0]) # [1, 1, 1]
    assert (2/3) == f1_2.compute(ground_truth, rec_list)

    ground_truth = np.asarray([1, 0, 0]) # [1, 0, 0]
    rec_list = np.asarray([1, 2]) # [0, 1, 1]
    assert 0 == f1_2.compute(ground_truth, rec_list)