# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from ..data import MatrixTrainSet, TestSet
from ..utils.common import validate_format
from ..metrics.rating import RatingMetric
from ..metrics.ranking import RankingMetric
from collections import OrderedDict
import numpy as np


class BaseMethod:
    """Base Evaluation Method

    Parameters
    ----------
    data: array-like
        The original data.

    data_format: str, default: 'UIR'
        The format of given data.

    test_set: :obj:`TestSet<cornac.data.TestSet>`, optional, default: None
        The test data.

    total_users: int, optional, default: None
        Total number of unique users in the data including train, val, and test sets

    total_users: int, optional, default: None
        Total number of unique items in the data including train, val, and test sets

    rating_threshold: float, optional, default: 1
        The minimum value that is considered to be a good rating used for ranking, \
        e.g, if the ratings are in {1, ..., 5}, then good_rating = 4.

    exclude_unknowns: bool, optional, default: False
        Ignore unknown users and items (cold-start) during evaluation and testing

    verbose: bool, optional, default: False
        Output running log
    """

    def __init__(self, data=None, data_format='UIR', train_set=None, test_set=None,
                 total_users=None, total_items=None, rating_threshold=1.0,
                 exclude_unknowns=False, verbose=False):
        self._data = data
        self.data_format = validate_format(data_format, self.valid_data_formats)
        self.train_set = train_set
        self.test_set = test_set
        self.total_users = total_users
        self.total_items = total_items
        self.rating_threshold = rating_threshold
        self.exclude_unknowns = exclude_unknowns
        self.verbose = verbose

        if verbose:
            print('rating_threshold = {:.1f}'.format(rating_threshold))
            print('exclude_unknowns = {}'.format(exclude_unknowns))

    @property
    def valid_data_formats(self):
        return ['UIR', 'UIRT']

    def _organize_metrics(self, metrics):
        """Organize metrics according to their types (rating or raking)

        Parameters
        ----------
        metrics: :obj:`iterable`
            List of metrics.

        """
        if isinstance(metrics, dict):
            rating_metrics = metrics.get('rating', [])
            ranking_metrics = metrics.get('ranking', [])
        elif isinstance(metrics, list):
            rating_metrics = [mt for mt in metrics if isinstance(mt, RatingMetric)]
            ranking_metrics = [mt for mt in metrics if isinstance(mt, RankingMetric)]
        else:
            raise ValueError('Type of metrics has to be either dict or list!')

        return rating_metrics, ranking_metrics

    def evaluate(self, model, metrics, user_based):
        """Evaluate given models according to given metrics

        Parameters
        ----------
        model: :obj:`cornac.models.Recommender`
            Recommender model to be evaluated.

        metrics: :obj:`iterable`
            List of metrics.

        user_based: bool
            Evaluation mode. Whether results are averaging based on number of users or number of ratings.

        """
        rating_metrics, ranking_metrics = self._organize_metrics(metrics)

        if self.train_set is None:
            raise ValueError('train_set is required but None!')

        if self.test_set is None:
            raise ValueError('test_set is required but None!')

        if self.total_users is None:
            self.total_users = len(set(self.train_set.get_uid_list() + self.test_set.get_uid_list()))

        if self.total_items is None:
            self.total_items = len(set(self.train_set.get_iid_list() + self.test_set.get_iid_list()))

        if self.verbose:
            print("Training started!")

        model.fit(self.train_set)

        if self.verbose:
            print("Evaluation started!")

        all_rating_gts = []
        all_rating_pds = []
        metric_user_results = {}
        for mt in (rating_metrics + ranking_metrics):
            metric_user_results[mt.name] = {}

        num_eval_users = len(self.test_set.get_users())
        for i, user_id in enumerate(self.test_set.get_users()):
            if self.verbose:
                if i % 1000 == 0 or (i + 1) == num_eval_users:
                    print(i, "users evaluated")

            # ignore unknown users when self.exclude_unknown
            if self.exclude_unknowns and self.train_set.is_unk_user(user_id):
                continue

            u_rating_gts = []
            u_rating_pds = []
            if self.exclude_unknowns:
                u_ranking_gts = np.zeros(self.train_set.num_items)
                candidate_item_ids = None  # all known items
            else:
                u_ranking_gts = np.zeros(self.total_items)
                candidate_item_ids = np.arange(self.total_items)

            for item_id, rating in self.test_set.get_ratings(user_id):
                # ignore unknown items when self.exclude_unknown
                if self.exclude_unknowns and self.train_set.is_unk_item(item_id):
                    continue

                if len(rating_metrics) > 0:
                    all_rating_gts.append(rating)
                    u_rating_gts.append(rating)

                    rating_pred = model.rate(user_id, item_id)
                    all_rating_pds.append(rating_pred)
                    u_rating_pds.append(rating_pred)

                # constructing ranking ground-truth
                if rating >= self.rating_threshold:
                    u_ranking_gts[item_id] = 1

            # per user evaluation of rating metrics
            if len(rating_metrics) > 0 and len(u_rating_gts) > 0:
                for mt in rating_metrics:
                    mt_score = mt.compute(np.asarray(u_rating_gts), np.asarray(u_rating_pds))
                    metric_user_results[mt.name][user_id] = mt_score

            # evaluation of ranking metrics
            if len(ranking_metrics) > 0 and u_ranking_gts.sum() > 0:
                u_ranking_pds = model.rank(user_id, candidate_item_ids)
                for mt in ranking_metrics:
                    mt_score = mt.compute(u_ranking_gts, u_ranking_pds)
                    metric_user_results[mt.name][user_id] = mt_score

        metric_avg_results = {}

        # avg results of rating metrics
        for mt in rating_metrics:
            if user_based:  # averaging over users
                user_results = list(metric_user_results[mt.name].values())
                metric_avg_results[mt.name] = np.asarray(user_results).mean()
            else:  # averaging over ratings
                metric_avg_results[mt.name] = mt.compute(np.asarray(all_rating_gts), np.asarray(all_rating_pds))

        # avg results of ranking metrics
        for mt in ranking_metrics:
            user_results = list(metric_user_results[mt.name].values())
            metric_avg_results[mt.name] = np.asarray(user_results).mean()

        return metric_avg_results, metric_user_results

    def _build_uir(self, train_data, test_data, val_data=None):
        global_uid_map = OrderedDict()
        global_iid_map = OrderedDict()
        global_ui_set = set()  # avoid duplicate ratings in the data

        if train_data is None:
            raise ValueError('train_data is required but None!')

        if test_data is None:
            raise ValueError('test_data is required but None!')

        if self.verbose:
            print('Building training set')
        self.train_set = MatrixTrainSet.from_uir(
            train_data, global_uid_map, global_iid_map, global_ui_set, self.verbose)

        if self.verbose:
            print('Building test set')
        self.test_set = TestSet.from_uir(
            test_data, global_uid_map, global_iid_map, global_ui_set, self.verbose)

        if not val_data is None:
            if self.verbose:
                print('Building validation set')
            self.val_set = TestSet.from_uir(
                val_data, global_uid_map, global_iid_map, global_ui_set, self.verbose)

        self.total_users = len(global_uid_map)
        self.total_items = len(global_iid_map)

    @classmethod
    def from_provided(cls, train_data, test_data, val_data=None, data_format='UIR',
                      rating_threshold=1.0, exclude_unknowns=False, verbose=False):
        """Constructing evaluation method given data.

        Parameters
        ----------
        train_data: array-like
            Training data

        test_data: array-like
            Test data

        val_data: array-like
            Validation data

        data_format: str, default: 'UIR'
            The format of given data.

        rating_threshold: float, default: 1.0
            Threshold to decide positive or negative preferences.

        exclude_unknowns: bool, default: False
            Whether to exclude unknown users/items in evaluation.

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        method: :obj:`<cornac.eval_methods.BaseMethod>`
            Evaluation method object.

        """
        method = cls(data_format=data_format, rating_threshold=rating_threshold,
                     exclude_unknowns=exclude_unknowns, verbose=verbose)
        if method.data_format == 'UIR':
            method._build_uir(train_data, test_data, val_data)

        return method
