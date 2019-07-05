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

from collections import OrderedDict, defaultdict
import time

import numpy as np
import tqdm

from ..data import TextModule
from ..data import ImageModule
from ..data import GraphModule
from ..data import MultimodalTrainSet
from ..data import MultimodalTestSet
from ..utils.common import validate_format
from ..metrics.rating import RatingMetric
from ..metrics.ranking import RankingMetric
from ..experiment.result import Result

VALID_DATA_FORMATS = ['UIR', 'UIRT']


class BaseMethod:
    """Base Evaluation Method

    Parameters
    ----------
    data: array-like
        The original data.

    data_format: str, default: 'UIR'
        The format of given data.

    total_users: int, optional, default: None
        Total number of unique users in the data including train, val, and test sets.

    total_users: int, optional, default: None
        Total number of unique items in the data including train, val, and test sets.

    rating_threshold: float, optional, default: 1.0
        The threshold to convert ratings into positive or negative feedback for ranking metrics.

    exclude_unknowns: bool, optional, default: False
        Ignore unknown users and items (cold-start) during evaluation.

    verbose: bool, optional, default: False
        Output running log
    """

    def __init__(self, data=None,
                 fmt='UIR',
                 rating_threshold=1.0,
                 exclude_unknowns=False,
                 verbose=False,
                 **kwargs):
        self._data = data
        self.data_format = validate_format(fmt, VALID_DATA_FORMATS)
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.rating_threshold = rating_threshold
        self.exclude_unknowns = exclude_unknowns
        self.verbose = verbose
        self.global_uid_map = OrderedDict()
        self.global_iid_map = OrderedDict()

        self.user_text = kwargs.get('user_text', None)
        self.user_image = kwargs.get('user_image', None)
        self.user_graph = kwargs.get('user_graph', None)
        self.item_text = kwargs.get('item_text', None)
        self.item_image = kwargs.get('item_image', None)
        self.item_graph = kwargs.get('item_graph', None)

        if verbose:
            print('rating_threshold = {:.1f}'.format(rating_threshold))
            print('exclude_unknowns = {}'.format(exclude_unknowns))

    @property
    def total_users(self):
        return len(self.global_uid_map)

    @property
    def total_items(self):
        return len(self.global_iid_map)

    @property
    def user_text(self):
        return self.__user_text

    @user_text.setter
    def user_text(self, input_module):
        if input_module is not None and not isinstance(input_module, TextModule):
            raise ValueError('input_module has to be instance of TextModule but {}'.format(type(input_module)))
        self.__user_text = input_module

    @property
    def user_image(self):
        return self.__user_image

    @user_image.setter
    def user_image(self, input_module):
        if input_module is not None and not isinstance(input_module, ImageModule):
            raise ValueError('input_module has to be instance of ImageModule but {}'.format(type(input_module)))
        self.__user_image = input_module

    @property
    def user_graph(self):
        return self.__user_graph

    @user_graph.setter
    def user_graph(self, input_module):
        if input_module is not None and not isinstance(input_module, GraphModule):
            raise ValueError('input_module has to be instance of GraphModule but {}'.format(type(input_module)))
        self.__user_graph = input_module

    @property
    def item_text(self):
        return self.__item_text

    @item_text.setter
    def item_text(self, input_module):
        if input_module is not None and not isinstance(input_module, TextModule):
            raise ValueError('input_module has to be instance of TextModule but {}'.format(type(input_module)))
        self.__item_text = input_module

    @property
    def item_image(self):
        return self.__item_image

    @item_image.setter
    def item_image(self, input_module):
        if input_module is not None and not isinstance(input_module, ImageModule):
            raise ValueError('input_module has to be instance of ImageModule but {}'.format(type(input_module)))
        self.__item_image = input_module

    @property
    def item_graph(self):
        return self.__item_graph

    @item_graph.setter
    def item_graph(self, input_module):
        if input_module is not None and not isinstance(input_module, GraphModule):
            raise ValueError('input_module has to be instance of GraphModule but {}'.format(type(input_module)))
        self.__item_graph = input_module

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

    def _build_uir(self, train_data, test_data, val_data=None):
        if train_data is None:
            raise ValueError('train_data is required but None!')
        if test_data is None:
            raise ValueError('test_data is required but None!')

        global_ui_set = set()  # avoid duplicate ratings in the data

        if self.verbose:
            print('Building training set')
        self.train_set = MultimodalTrainSet.from_uir(
            train_data, self.global_uid_map, self.global_iid_map, global_ui_set, self.verbose)

        if self.verbose:
            print('Building test set')
        self.test_set = MultimodalTestSet.from_uir(
            test_data, self.global_uid_map, self.global_iid_map, global_ui_set, self.verbose)

        if val_data is not None:
            if self.verbose:
                print('Building validation set')
            self.val_set = MultimodalTestSet.from_uir(
                val_data, self.global_uid_map, self.global_iid_map, global_ui_set, self.verbose)

    def _build_modules(self):
        for user_module in [self.user_text, self.user_image, self.user_graph]:
            if user_module is None:
                continue
            user_module.build(id_map=self.global_uid_map)

        for item_module in [self.item_text, self.item_image, self.item_graph]:
            if item_module is None:
                continue
            item_module.build(id_map=self.global_iid_map)

        for data_set in [self.train_set, self.test_set, self.val_set]:
            if data_set is None:
                continue
            data_set.add_modules(user_text=self.user_text,
                                 user_image=self.user_image,
                                 user_graph=self.user_graph,
                                 item_text=self.item_text,
                                 item_image=self.item_image,
                                 item_graph=self.item_graph)

    def build(self, train_data, test_data, val_data=None):
        self.global_uid_map.clear()
        self.global_iid_map.clear()

        if self.data_format == 'UIR':
            self._build_uir(train_data, test_data, val_data)

        self._build_modules()

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

        if self.verbose:
            print('\n[{}] Training started!'.format(model.name))
        start = time.time()
        model.fit(self.train_set)
        train_time = time.time() - start

        if self.verbose:
            print('\n[{}] Evaluation started!'.format(model.name))
        start = time.time()

        all_pd_ratings = []
        all_gt_ratings = []

        metric_user_results = defaultdict()
        for mt in (rating_metrics + ranking_metrics):
            metric_user_results[mt.name] = {}

        for user_id in tqdm.tqdm(self.test_set.users, disable=not self.verbose):
            # ignore unknown users when self.exclude_unknown
            if self.exclude_unknowns and self.train_set.is_unk_user(user_id):
                continue

            u_pd_ratings = []
            u_gt_ratings = []

            if self.exclude_unknowns:
                u_gt_pos = np.zeros(self.train_set.num_items, dtype=np.int)
                item_ids = None  # all known items
            else:
                u_gt_pos = np.zeros(self.total_items, dtype=np.int)
                item_ids = np.arange(self.total_items)

            u_gt_neg = np.ones_like(u_gt_pos, dtype=np.int)
            if not self.train_set.is_unk_user(user_id):
                u_train_ratings = self.train_set.matrix[user_id].A.ravel()
                u_train_neg = np.where(u_train_ratings < self.rating_threshold, 1, 0)
                u_gt_neg[:len(u_train_neg)] = u_train_neg

            for item_id, rating in self.test_set.get_ratings(user_id):
                # ignore unknown items when self.exclude_unknown
                if self.exclude_unknowns and self.train_set.is_unk_item(item_id):
                    continue

                if len(rating_metrics) > 0:
                    all_gt_ratings.append(rating)
                    u_gt_ratings.append(rating)

                    rating_pred = model.rate(user_id, item_id)
                    all_pd_ratings.append(rating_pred)
                    u_pd_ratings.append(rating_pred)

                # constructing ranking ground-truth
                if rating >= self.rating_threshold:
                    u_gt_pos[item_id] = 1
                    u_gt_neg[item_id] = 0

            # per user evaluation of rating metrics
            if len(rating_metrics) > 0 and len(u_gt_ratings) > 0:
                for mt in rating_metrics:
                    mt_score = mt.compute(gt_ratings=np.asarray(u_gt_ratings),
                                          pd_ratings=np.asarray(u_pd_ratings))
                    metric_user_results[mt.name][user_id] = mt_score

            # evaluation of ranking metrics
            if len(ranking_metrics) > 0 and u_gt_pos.sum() > 0:
                item_rank, item_scores = model.rank(user_id, item_ids)
                for mt in ranking_metrics:
                    mt_score = mt.compute(gt_pos=u_gt_pos,
                                          gt_neg=u_gt_neg,
                                          pd_rank=item_rank,
                                          pd_scores=item_scores)
                    metric_user_results[mt.name][user_id] = mt_score

        metric_avg_results = OrderedDict()

        # avg results of rating metrics
        for mt in rating_metrics:
            if user_based:  # averaging over users
                user_results = list(metric_user_results[mt.name].values())
                metric_avg_results[mt.name] = np.mean(user_results)
            else:  # averaging over ratings
                metric_avg_results[mt.name] = mt.compute(gt_ratings=np.asarray(all_gt_ratings),
                                                         pd_ratings=np.asarray(all_pd_ratings))

        # avg results of ranking metrics
        for mt in ranking_metrics:
            user_results = list(metric_user_results[mt.name].values())
            metric_avg_results[mt.name] = np.mean(user_results)

        test_time = time.time() - start

        metric_avg_results['Train (s)'] = train_time
        metric_avg_results['Test (s)'] = test_time

        return Result(model.name, metric_avg_results, metric_user_results)

    @classmethod
    def from_splits(cls, train_data, test_data, val_data=None, data_format='UIR',
                    rating_threshold=1.0, exclude_unknowns=False, verbose=False, **kwargs):
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
                     exclude_unknowns=exclude_unknowns, verbose=verbose, **kwargs)
        method.build(train_data=train_data, test_data=test_data, val_data=val_data)
        return method
