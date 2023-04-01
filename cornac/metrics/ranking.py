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

import numpy as np
from scipy.stats import rankdata


class RankingMetric:
    """Ranking Metric.

    Attributes
    ----------
    type: string, value: 'ranking'
        Type of the metric, e.g., "ranking", "rating".

    name: string, default: None
        Name of the measure.

    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    """

    def __init__(self, name=None, k=-1, higher_better=True):
        assert hasattr(k, "__len__") or k == -1 or k > 0

        self.type = "ranking"
        self.name = name
        self.k = k
        self.higher_better = higher_better

    def compute(self, **kwargs):
        raise NotImplementedError()


class NDCG(RankingMetric):
    """Normalized Discount Cumulative Gain.

    Parameters
    ----------
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    References
    ----------
    https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    """

    def __init__(self, k=-1):
        RankingMetric.__init__(self, name="NDCG@{}".format(k), k=k)

    @staticmethod
    def dcg_score(gt_pos, pd_rank, k=-1):
        """Compute Discounted Cumulative Gain score.

        Parameters
        ----------
        gt_pos: Numpy array
            Binary vector of positive items.

        pd_rank: Numpy array
            Item ranking prediction.

        k: int, optional, default: -1 (all)
            The number of items in the top@k list.
            If None, all items will be considered.

        Returns
        -------
        dcg: A scalar
            Discounted Cumulative Gain score.

        """
        if k > 0:
            truncated_pd_rank = pd_rank[:k]
        else:
            truncated_pd_rank = pd_rank

        ranked_scores = np.take(gt_pos, truncated_pd_rank)
        gain = 2 ** ranked_scores - 1
        discounts = np.log2(np.arange(len(ranked_scores)) + 2)

        return np.sum(gain / discounts)

    def compute(self, gt_pos, pd_rank, **kwargs):
        """Compute Normalized Discounted Cumulative Gain score.

        Parameters
        ----------
        gt_pos: Numpy array
            Binary vector of positive items.

        pd_rank: Numpy array
            Item ranking prediction.

        **kwargs: For compatibility

        Returns
        -------
        ndcg: A scalar
            Normalized Discounted Cumulative Gain score.

        """
        dcg = self.dcg_score(gt_pos, pd_rank, self.k)
        idcg = self.dcg_score(gt_pos, np.argsort(gt_pos)[::-1], self.k)
        ndcg = dcg / idcg

        return ndcg


class NCRR(RankingMetric):
    """Normalized Cumulative Reciprocal Rank.

    Parameters
    ----------
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    """

    def __init__(self, k=-1):
        RankingMetric.__init__(self, name="NCRR@{}".format(k), k=k)

    def compute(self, gt_pos, pd_rank, **kwargs):
        """Compute Normalized Cumulative Reciprocal Rank score.

        Parameters
        ----------
        gt_pos: Numpy array
            Binary vector of positive items.

        pd_rank: Numpy array
            Item ranking prediction.

        **kwargs: For compatibility

        Returns
        -------
        ncrr: A scalar
            Normalized Cumulative Reciprocal Rank score.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[: self.k]
        else:
            truncated_pd_rank = pd_rank

        gt_pos_items = np.nonzero(gt_pos > 0)

        # Compute CRR
        rec_rank = np.where(np.in1d(truncated_pd_rank, gt_pos_items))[0]
        if len(rec_rank) == 0:
            return 0.0
        rec_rank = rec_rank + 1  # +1 because indices starts from 0 in python
        crr = np.sum(1.0 / rec_rank)

        # Compute Ideal CRR
        max_nb_pos = min(len(gt_pos_items[0]), len(truncated_pd_rank))
        ideal_rank = np.arange(max_nb_pos)
        ideal_rank = ideal_rank + 1  # +1 because indices starts from 0 in python
        icrr = np.sum(1.0 / ideal_rank)

        # Compute nDCG
        ncrr_i = crr / icrr

        return ncrr_i


class MRR(RankingMetric):
    """Mean Reciprocal Rank.

    References
    ----------
    https://en.wikipedia.org/wiki/Mean_reciprocal_rank
    """

    def __init__(self):
        RankingMetric.__init__(self, name="MRR")

    def compute(self, gt_pos, pd_rank, **kwargs):
        """Compute Mean Reciprocal Rank score.

        Parameters
        ----------
        gt_pos: Numpy array
            Binary vector of positive items.

        pd_rank: Numpy array
            Item ranking prediction.

        **kwargs: For compatibility

        Returns
        -------
        mrr: A scalar
            Mean Reciprocal Rank score.

        """
        gt_pos_items = np.nonzero(gt_pos > 0)
        matched_items = np.nonzero(np.in1d(pd_rank, gt_pos_items))[0]

        if len(matched_items) == 0:
            raise ValueError(
                "No matched between ground-truth items and recommendations"
            )

        mrr = np.divide(
            1, (matched_items[0] + 1)
        )  # +1 because indices start from 0 in python
        return mrr


class MeasureAtK(RankingMetric):
    """Measure at K.

    Attributes
    ----------
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    """

    def __init__(self, name=None, k=-1):
        RankingMetric.__init__(self, name, k)

    def compute(self, gt_pos, pd_rank, **kwargs):
        """Compute TP, TP+FN, and TP+FP.

        Parameters
        ----------
        gt_pos: Numpy array
            Binary vector of positive items.

        pd_rank: Numpy array
            Item ranking prediction.

        **kwargs: For compatibility

        Returns
        -------
        tp: A scalar
            True positive.

        tp_fn: A scalar
            True positive + false negative.

        tp_fp: A scalar
            True positive + false positive.

        """
        if self.k > 0:
            truncated_pd_rank = pd_rank[: self.k]
        else:
            truncated_pd_rank = pd_rank

        pred = np.zeros_like(gt_pos)
        pred[truncated_pd_rank] = 1

        tp = np.sum(pred * gt_pos)
        tp_fn = np.sum(gt_pos)
        tp_fp = np.sum(pred)

        return tp, tp_fn, tp_fp


class Precision(MeasureAtK):
    """Precision@K.

    Parameters
    ----------
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    """

    def __init__(self, k=-1):
        super().__init__(name="Precision@{}".format(k), k=k)

    def compute(self, gt_pos, pd_rank, **kwargs):
        """Compute Precision score.

        Parameters
        ----------
        gt_pos: Numpy array
            Binary vector of positive items.

        pd_rank: Numpy array
            Item ranking prediction.

        **kwargs: For compatibility

        Returns
        -------
        res: A scalar
            Precision score.

        """
        tp, _, tp_fp = MeasureAtK.compute(self, gt_pos, pd_rank, **kwargs)
        return tp / tp_fp


class Recall(MeasureAtK):
    """Recall@K.

    Parameters
    ----------
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    """

    def __init__(self, k=-1):
        super().__init__(name="Recall@{}".format(k), k=k)

    def compute(self, gt_pos, pd_rank, **kwargs):
        """Compute Recall score.

        Parameters
        ----------
        gt_pos: Numpy array
            Binary vector of positive items.

        pd_rank: Numpy array
            Item ranking prediction.

        **kwargs: For compatibility

        Returns
        -------
        res: A scalar
            Recall score.

        """
        tp, tp_fn, _ = MeasureAtK.compute(self, gt_pos, pd_rank, **kwargs)
        return tp / tp_fn


class FMeasure(MeasureAtK):
    """F-measure@K.

    Parameters
    ----------
    k: int or list, optional, default: -1 (all)
        The number of items in the top@k list.
        If None, all items will be considered.

    """

    def __init__(self, k=-1):
        super().__init__(name="F1@{}".format(k), k=k)

    def compute(self, gt_pos, pd_rank, **kwargs):
        """Compute F-Measure.

        Parameters
        ----------
        gt_pos: Numpy array
            Binary vector of positive items.

        pd_rank: Numpy array
            Item ranking prediction.

        **kwargs: For compatibility

        Returns
        -------
        res: A scalar
            F-Measure score.

        """
        tp, tp_fn, tp_fp = MeasureAtK.compute(self, gt_pos, pd_rank, **kwargs)

        prec = tp / tp_fp
        rec = tp / tp_fn
        if (prec + rec) > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0

        return f1


class AUC(RankingMetric):
    """Area Under the ROC Curve (AUC).

    References
    ----------
    https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf

    """

    def __init__(self):
        RankingMetric.__init__(self, name="AUC")

    def compute(self, pd_scores, gt_pos, gt_neg=None, **kwargs):
        """Compute Area Under the ROC Curve (AUC).

        Parameters
        ----------
        pd_scores: Numpy array
            Prediction scores for items.

        gt_pos: Numpy array
            Binary vector of positive items.

        gt_neg: Numpy array, optional
            Binary vector of negative items.
            If None, negation of gt_pos will be used.

        **kwargs: For compatibility

        Returns
        -------
        res: A scalar
            AUC score.

        """
        if gt_neg is None:
            gt_neg = np.logical_not(gt_pos)

        pos_scores = pd_scores[gt_pos.astype('bool')]
        neg_scores = pd_scores[gt_neg.astype('bool')]
        ui_scores = np.repeat(pos_scores, len(neg_scores))
        uj_scores = np.tile(neg_scores, len(pos_scores))

        return (ui_scores > uj_scores).sum() / len(uj_scores)


class MAP(RankingMetric):
    """Mean Average Precision (MAP).

    References
    ----------
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision

    """

    def __init__(self):
        RankingMetric.__init__(self, name="MAP")

    def compute(self, pd_scores, gt_pos, **kwargs):
        """Compute Average Precision.

        Parameters
        ----------
        pd_scores: Numpy array
            Prediction scores for items.

        gt_pos: Numpy array
            Binary vector of positive items.

        **kwargs: For compatibility

        Returns
        -------
        res: A scalar
            AP score.

        """
        relevant = gt_pos.astype('bool')
        rank = rankdata(-pd_scores, "max")[relevant]
        L = rankdata(-pd_scores[relevant], "max")
        ans = (L / rank).mean()

        return ans
