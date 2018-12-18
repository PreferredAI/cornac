# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np

def test_RankingMetric():
    from ..ranking import RankingMetric
    metric = RankingMetric()

    assert metric.type == 'ranking'
    assert metric.name is None

    try:
        metric.compute(None, None)
    except NotImplementedError:
        assert True


def test_NDCG():
    from ..ranking import NDCG
    ndcg = NDCG()

    assert ndcg.type == 'ranking'
    assert ndcg.name == 'NDCG'

    assert 1 == ndcg.compute(np.asarray([1]), np.asarray([0]))

    ground_truth = np.asarray([1, 0, 1]) # [1, 3, 2]
    rec_list = np.asarray([0, 2, 1]) # [1, 3, 2]
    assert 1 == ndcg.compute(ground_truth, rec_list)

    ndcg_2 = NDCG(k=2)

    ground_truth = np.asarray([0, 0, 1]) # [3, 1, 2]
    rec_list = np.asarray([1, 2, 0]) # [2, 3, 1]
    assert 0.63 == float('{:.2f}'.format(ndcg_2.compute(ground_truth, rec_list)))


# def test_MRR():
#     from ..ranking import MRR
#     mrr = MRR()
#
#     assert mrr.type == 'ranking'
#     assert mrr.name == 'MRR'
#
#     assert 1 == mrr.compute(np.asarray([1]), np.asarray([0]))
#
#     ground_truths = np.asarray([1, 0, 1]) # [1, 3, 2]
#     rec_list = np.asarray([0, 2, 1]) # [1, 3, 2]
#     assert 1 == mrr.compute(ground_truths, rec_list)