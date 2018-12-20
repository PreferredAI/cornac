# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah
         Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np


class RankingMetric:
    """Ranking Metric.

    Parameters
    ----------
    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking", "rating".
    """

    def __init__(self, name=None, k=-1):
        self.type = 'ranking'
        self.name = name
        self.k = k

    def compute(self, ground_truth, rec_list):
        raise NotImplementedError()


# todo: take into account 'm' parameter
class NDCG(RankingMetric):
    """Normalized Discount Cumulative Gain.

    Parameters
    ----------
    k: int, optional, default: -1 (all)
        The number of items in the top@k list, \
        if None then all items are considered to compute NDCG.

    name: string, value: 'NDCG'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".

    References
    ----------
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """

    def __init__(self, k=-1):
        RankingMetric.__init__(self, name='NDCG@{}'.format(k), k=k)


    @staticmethod
    def dcg_score(ground_truth, rec_list, k):
        if k > 0:
            rec_list = rec_list[:k]

        ground_truth = np.take(ground_truth, rec_list)

        gain = 2 ** ground_truth - 1
        discounts = np.log2(np.arange(len(ground_truth)) + 2)

        return np.sum(gain / discounts)

    # Compute nDCG
    def compute(self, ground_truth, rec_list):
        dcg = self.dcg_score(ground_truth, rec_list, self.k)
        idcg = self.dcg_score(ground_truth, np.argsort(ground_truth)[::-1], self.k)
        ndcg = dcg / idcg

        return ndcg


# todo: take into account 'm' parameter
class NCRR(RankingMetric):
    """Normalized Cumulative Reciprocal Rank.

    Parameters
    ----------
    k: int, optional, default: -1 (all)
        The number of items in the top@k list, \
        if None then all items are considered to compute NDCG.

    name: string, value: 'NCRR'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".
    """

    def __init__(self, k=-1):
        RankingMetric.__init__(self, name='NCRR@{}'.format(k), k=k)


    # Compute nCRR for a single user i
    def compute(self, ground_truth, rec_list):
        ground_truth_idx = np.nonzero(ground_truth > 0)

        # Compute Ideal CRR
        ideal_rank = np.arange(len(ground_truth_idx[0]))
        ideal_rank = ideal_rank + 1  # +1 because indices starts from 0 in python
        icrr = np.sum(1. / ideal_rank)

        # Compute CRR
        rec_rank = np.where(np.in1d(rec_list, ground_truth_idx))[0]
        rec_rank = rec_rank + 1  # +1 because indices starts from 0 in python
        crr = np.sum(1. / rec_rank)

        # Compute nDCG
        ncrr_i = crr / icrr

        return ncrr_i


class MRR(RankingMetric):
    """Mean Reciprocal Rank.

    Parameters
    ----------
    name: string, value: 'MRR'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".

    References
    ----------
    https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    """

    def __init__(self):
        RankingMetric.__init__(self, name='MRR')

    # Compute MRR for a single user i
    def compute(self, ground_truth, rec_list):
        ground_truth_idx = np.nonzero(ground_truth > 0)
        matched_indices = np.nonzero(np.in1d(rec_list, ground_truth_idx))[0]

        if len(matched_indices) == 0:
            raise ValueError('No matched between ground truth and recommended list')

        mrr = np.divide(1, (matched_indices[0] + 1))  # +1 because indices start from 0 in python
        return mrr


class MeasureAtK(RankingMetric):

    def __init__(self, name=None, k=-1):
        RankingMetric.__init__(self, name, k)
        self.tp = None
        self.tp_fn = None
        self.tp_fp = None

    # Evaluate TopK list for a single user: Precision@K, Recall@K, F-meansure@K (F1)
    def compute(self, ground_truth, rec_list):
        if self.k > 0:
            rec_list = rec_list[:self.k]

        # ground_truth_bin = np.zeros(len(ground_truth))
        # ground_truth_bin[np.nonzero(ground_truth > 0)] = 1
        ground_truth_bin = ground_truth # ground_truth assumed to be already a binary vector

        pred = np.zeros_like(ground_truth_bin)
        pred[rec_list] = 1

        self.tp = np.sum(pred * ground_truth_bin)
        self.tp_fn = np.sum(ground_truth_bin)
        self.tp_fp = np.sum(pred)


class Precision(MeasureAtK):
    """Precision@K.

    Parameters
    ----------
    k: int, optional, default: -1 (all)
        The number of items in the top@k list.
        
    name: string, value: 'Precision@k'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".
    """

    def __init__(self, k=-1):
        MeasureAtK.__init__(self, name="Precision@{}".format(k), k=k)

    # Compute Precision@K for a single user i
    def compute(self, ground_truth, rec_list):
        super().compute(ground_truth, rec_list)
        prec = self.tp / self.tp_fp
        return prec


class Recall(MeasureAtK):
    """Recall@K.

    Parameters
    ----------
    k: int, optional, default: -1 (all)
        The number of items in the top@k list.
        
    name: string, value: 'Recall@k'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".
    """

    def __init__(self, k=-1):
        MeasureAtK.__init__(self, name="Recall@{}".format(k), k=k)

    # Compute Precision@K for a single user i
    def compute(self, ground_truth, rec_list):
        super().compute(ground_truth, rec_list)
        rec = self.tp / self.tp_fn
        return rec


class FMeasure(MeasureAtK):
    """F-measure@K@.

    Parameters
    ----------
    k: int, optional, default: -1 (all)
        The number of items in the top@k list.
        
    name: string, value: 'F1@k'
        Name of the measure.

    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking".
    """

    def __init__(self, k=-1):
        MeasureAtK.__init__(self, name="F1@{}".format(k), k=k)

    # Compute Precision@K for a single user i
    def compute(self, ground_truth, rec_list):
        super().compute(ground_truth, rec_list)

        prec = self.tp / self.tp_fp
        rec = self.tp / self.tp_fn
        if (prec + rec):
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0
        return f1
