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

    total_users: int, optional, default: None
        Total number of unique users in the data including train, val, and test sets

    total_users: int, optional, default: None
        Total number of unique items in the data including train, val, and test sets

    rating_threshold: float, optional, default: 1
        The minimum value that is considered to be a good rating used for ranking, \
        e.g, if the ratings are in {1, ..., 5}, then good_rating = 4.

    include_unknowns: bool, optional, default: True
        Taking into account unknown users and items (cold-start) in the evaluation

    """

    def __init__(self, train_set=None, val_set=None, test_set=None,
                 total_users=None, total_items=None, rating_threshold=1., include_unknowns=True):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.total_users = total_users
        self.total_items = total_items
        self.rating_threshold = rating_threshold
        self.include_unknowns = include_unknowns
        print('Rating threshold = {:.1f}'.format(rating_threshold))
        print('Including unknowns = {}'.format(include_unknowns))


    def evaluate(self, model, metrics):
        rating_metrics = metrics.get('rating', [])
        ranking_metrics = metrics.get('ranking', [])

        if self.train_set is None:
            raise ValueError('train_set is required but None!')

        if self.test_set is None:
            raise ValueError('test_set is required but None!')

        if self.total_users is None:
            self.total_users = len(set(self.train_set.get_uid_list() +
                                       self.val_set.get_uid_list() +
                                       self.test_set.get_uid_list()))

        if self.total_items is None:
            self.total_items = len(set(self.train_set.get_iid_list() +
                                       self.val_set.get_iid_list() +
                                       self.test_set.get_iid_list()))

        print("Training started!")

        model.fit_(self.train_set)

        print("Evaluation started!")

        metric_avg_results = {}
        metric_user_results = {}

        rating_gts = []
        rating_pds = []
        for mt in (rating_metrics + ranking_metrics):
            metric_avg_results[mt.name] = []
            metric_user_results[mt.name] = {}

        for i, user_id in enumerate(self.test_set.get_users()):
            if i % 1000 == 0:
                print(i, "processed users")

            # ignore unknown users when self.include_unknowns=False
            if not self.train_set.is_known_user(user_id) and not self.include_unknowns:
                continue

            u_rating_gts = []
            u_rating_pds = []
            if self.include_unknowns:
                u_ranking_gts = np.zeros(self.total_items)
            else:
                u_ranking_gts = np.zeros(self.train_set.num_items)

            for item_id, rating in self.test_set.get_ratings(user_id):
                # ignore unknown items when self.include_unknowns=False
                if not self.train_set.is_known_item(item_id) and not self.include_unknowns:
                    continue

                rating_gts.append(rating)
                u_rating_gts.append(rating)
                if len(rating_metrics) > 0:
                    prediction = model.score_(user_id, item_id)
                    rating_pds.append(prediction)
                    u_rating_pds.append(prediction)

                # constructing ground-truth rank list
                if rating >= self.rating_threshold:
                    u_ranking_gts[item_id] = 1

            # per user evaluation for rating metrics
            if len(rating_metrics) > 0:
                u_rating_gts = np.asarray(u_rating_gts, dtype=np.float)
                u_rating_pds = np.asarray(u_rating_pds, dtype=np.float)
                for mt in rating_metrics:
                    mt_score = mt.compute(u_rating_gts, u_rating_pds)
                    metric_user_results[mt.name][user_id] = mt_score
                    metric_avg_results[mt.name].append(mt_score)

            if u_ranking_gts.sum() == 0: # no ranking ground-truth for this user
                continue

            # per user evaluation for ranking metrics
            if len(ranking_metrics) > 0:
                u_ranking_pds = model.rank_(user_id)
                for mt in ranking_metrics:
                    mt_score = mt.compute(u_ranking_gts, u_ranking_pds)
                    metric_user_results[mt.name][user_id] = mt_score
                    metric_avg_results[mt.name].append(mt_score)

        res_avg = []

        # avg results of ranking metrics
        for mt in ranking_metrics:
            metric_avg_results[mt.name] = np.asarray(metric_avg_results[mt.name]).mean()

            res_avg.append(metric_avg_results[mt.name])

        # avg results of rating metrics
        # rating_gts = np.asarray(rating_gts, dtype=np.float)
        # rating_pds = np.asarray(rating_pds, dtype=np.float)
        for mt in rating_metrics:
            # metric_avg_results[mt.name] = mt.compute(rating_gts, rating_pds)
            metric_avg_results[mt.name] = np.asarray(metric_avg_results[mt.name]).mean()

            res_avg.append(metric_avg_results[mt.name])

        # return metric_avg_results, metric_user_results

        res_tot = {"ResAvg": np.asarray(res_avg), "ResPerUser": metric_user_results}
        return res_tot