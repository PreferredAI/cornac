# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np


class BaseStrategy:
    """Base Evaluation Strategy

    Parameters
    ----------

    train_set: :obj:`TrainSet<cornac.data.TrainSet>`, optional, default: None
        The training data.

    val_set: :obj:`TestSet<cornac.data.TestSet>`, optional, default: None
        The validation data.

    test_set: :obj:`TestSet<cornac.data.TestSet>`, optional, default: None
        The test data.

    rating_threshold: float, optional, default: 1
        The minimum value that is considered to be a good rating used for ranking, \
        e.g, if the ratings are in {1, ..., 5}, then good_rating = 4.
    """

    def __init__(self, train_set=None, val_set=None, test_set=None, rating_threshold=1.):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.rating_threshold = rating_threshold

    def evaluate(self, model, metrics):
        ranking_metrics = []
        rating_metrics = []
        for m in metrics:
            if m.type == 'ranking':
                ranking_metrics.append(m)
            elif m.type == 'rating':
                rating_metrics.append(m)

        if self.train_set is None:
            raise ValueError('train_set is required but None!')

        if self.test_set is None:
            raise ValueError('test_set is required but None!')

        model.fit(self.train_set.matrix)

        print("Starting evaluation")

        metric_avg_results = {}
        metric_user_results = {}

        rating_gts = []
        rating_pds = []
        for m in ranking_metrics:
            metric_avg_results[m.name] = []
            metric_user_results[m.name] = {}

        for user in self.test_set.get_users():
            u_rating_gts = []
            u_rating_pds = []
            u_ranking_gts = np.zeros(self.train_set.num_items)

            for item, rating in self.test_set.get_ratings(user):
                rating_gts.append(rating)
                u_rating_gts.append(rating)
                if len(rating_metrics) > 0:
                    prediction = model.score(user, item)
                    rating_pds.append(prediction)
                    u_rating_pds.append(prediction)

                # constructing ground-truth rank list
                if self.train_set.is_known_item(item) and rating >= self.rating_threshold:
                    mapped_iid = self.train_set.map_iid[item]
                    u_ranking_gts[mapped_iid] = 1

            # per user evaluation for rating metrics
            if len(rating_metrics) > 0:
                u_rating_gts = np.asarray(u_rating_gts, dtype=np.float)
                u_rating_pds = np.asarray(u_rating_pds, dtype=np.float)
                for m in rating_metrics:
                    metric_user_results[m.name][user] = m.compute(u_rating_gts, u_rating_pds)

            # per user evaluation for ranking metrics
            if len(ranking_metrics) > 0:
                u_ranking_pds = model.rank(user)
                for m in ranking_metrics:
                    metric_user_results[m.name][user] = m.compute(u_ranking_gts, u_ranking_pds)

        # avg results of rating metrics
        rating_gts = np.asarray(rating_gts, dtype=np.float)
        rating_pds = np.asarray(rating_pds, dtype=np.float)
        for m in rating_metrics:
            metric_avg_results[m.name] = m.compute(rating_gts, rating_pds)

        # avg results of ranking metrics
        for m in ranking_metrics:
            metric_avg_results[m.name] = np.asarray(metric_avg_results[m.name]).mean()

        return metric_avg_results, metric_user_results
