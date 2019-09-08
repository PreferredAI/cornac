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
from scipy.sparse import csr_matrix

from ..data import TextModality
from ..data import ImageModality
from ..data import GraphModality
from ..data import Dataset
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

    seed: int, optional, default: None
        Random seed for reproduce the splitting.

    exclude_unknowns: bool, optional, default: False
        Ignore unknown users and items (cold-start) during evaluation.

    verbose: bool, optional, default: False
        Output running log

    """

    def __init__(self, data=None,
                 fmt='UIR',
                 rating_threshold=1.0,
                 seed=None,
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
        self.seed = seed
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
    def user_text(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, TextModality):
            raise ValueError('input_modality has to be instance of TextModality but {}'.format(type(input_modality)))
        self.__user_text = input_modality

    @property
    def user_image(self):
        return self.__user_image

    @user_image.setter
    def user_image(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, ImageModality):
            raise ValueError('input_modality has to be instance of ImageModality but {}'.format(type(input_modality)))
        self.__user_image = input_modality

    @property
    def user_graph(self):
        return self.__user_graph

    @user_graph.setter
    def user_graph(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, GraphModality):
            raise ValueError('input_modality has to be instance of GraphModality but {}'.format(type(input_modality)))
        self.__user_graph = input_modality

    @property
    def item_text(self):
        return self.__item_text

    @item_text.setter
    def item_text(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, TextModality):
            raise ValueError('input_modality has to be instance of TextModality but {}'.format(type(input_modality)))
        self.__item_text = input_modality

    @property
    def item_image(self):
        return self.__item_image

    @item_image.setter
    def item_image(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, ImageModality):
            raise ValueError('input_modality has to be instance of ImageModality but {}'.format(type(input_modality)))
        self.__item_image = input_modality

    @property
    def item_graph(self):
        return self.__item_graph

    @item_graph.setter
    def item_graph(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, GraphModality):
            raise ValueError('input_modality has to be instance of GraphModality but {}'.format(type(input_modality)))
        self.__item_graph = input_modality

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
        self.train_set = Dataset.from_uir(data=train_data,
                                          global_uid_map=self.global_uid_map, global_iid_map=self.global_iid_map,
                                          seed=self.seed, exclude_unknowns=False)
        if self.verbose:
            print('---')
            print('Training data:')
            print('Number of users = {}'.format(self.train_set.num_users))
            print('Number of items = {}'.format(self.train_set.num_items))
            print('Max rating = {:.1f}'.format(self.train_set.max_rating))
            print('Min rating = {:.1f}'.format(self.train_set.min_rating))
            print('Global mean = {:.1f}'.format(self.train_set.global_mean))

        self.test_set = Dataset.from_uir(data=test_data,
                                         global_uid_map=self.global_uid_map, global_iid_map=self.global_iid_map,
                                         seed=self.seed, exclude_unknowns=self.exclude_unknowns)
        if self.verbose:
            print('---')
            print('Test data:')
            print('Number of users = {}'.format(len(self.test_set.uid_map)))
            print('Number of items = {}'.format(len(self.test_set.iid_map)))
            print('Number of unknown users = {}'.format(self.test_set.num_users - self.train_set.num_users))
            print('Number of unknown items = {}'.format(self.test_set.num_items - self.train_set.num_items))

        if val_data is not None and len(val_data) > 0:
            self.val_set = Dataset.from_uir(data=val_data,
                                            global_uid_map=self.global_uid_map, global_iid_map=self.global_iid_map,
                                            seed=self.seed, exclude_unknowns=True)
            if self.verbose:
                print('---')
                print('Validation data:')
                print('Number of users = {}'.format(len(self.val_set.uid_map)))
                print('Number of items = {}'.format(len(self.val_set.iid_map)))

    def _build_modalities(self):
        for user_modality in [self.user_text, self.user_image, self.user_graph]:
            if user_modality is None:
                continue
            user_modality.build(id_map=self.global_uid_map)

        for item_modality in [self.item_text, self.item_image, self.item_graph]:
            if item_modality is None:
                continue
            item_modality.build(id_map=self.global_iid_map)

        for data_set in [self.train_set, self.test_set, self.val_set]:
            if data_set is None:
                continue
            data_set.add_modalities(user_text=self.user_text,
                                    user_image=self.user_image,
                                    user_graph=self.user_graph,
                                    item_text=self.item_text,
                                    item_image=self.item_image,
                                    item_graph=self.item_graph)

    def build(self, train_data, test_data, val_data=None):
        if train_data is None or len(train_data) == 0:
            raise ValueError('train_data is required but None or empty!')
        if test_data is None or len(test_data) == 0:
            raise ValueError('test_data is required but None or empty!')

        self.global_uid_map.clear()
        self.global_iid_map.clear()

        if self.data_format == 'UIR':
            self._build_uir(train_data, test_data, val_data)

        self._build_modalities()

    def _rating_eval(self, model, metrics, metric_avg_results, metric_user_results, user_based):
        """Evaluate model on provided rating metrics
        """

        if len(metrics) == 0:
            return

        (u_indices, i_indices, r_values) = self.test_set.uir_tuple
        r_preds = np.fromiter((model.rate(user_idx, item_idx).item()
                               for user_idx, item_idx in zip(u_indices, i_indices)),
                              dtype=np.float, count=len(u_indices))

        gt_mat = self.test_set.csr_matrix
        pd_mat = csr_matrix((r_preds, (u_indices, i_indices)), shape=gt_mat.shape)

        for mt in metrics:
            if user_based:  # averaging over users
                metric_user_results[mt.name] = {user_idx: mt.compute(gt_ratings=gt_mat.getrow(user_idx).data,
                                                                     pd_ratings=pd_mat.getrow(user_idx).data).item()
                                                for user_idx in self.test_set.user_indices}
                metric_avg_results[mt.name] = sum(metric_user_results[mt.name].values()) / \
                                              len(metric_user_results[mt.name])
            else:  # averaging over ratings
                metric_avg_results[mt.name] = mt.compute(gt_ratings=r_values, pd_ratings=r_preds)

    def _ranking_eval(self, model, metrics, metric_avg_results, metric_user_results):
        """Evaluate model on provided ranking metrics
        """

        if len(metrics) == 0:
            return

        gt_mat = self.test_set.csr_matrix
        train_mat = self.train_set.csr_matrix
        val_mat = None if self.val_set is None else self.val_set.csr_matrix

        def pos_items(csr_row):
            return [item_idx
                    for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
                    if rating >= self.rating_threshold]

        for user_idx in tqdm.tqdm(self.test_set.user_indices, disable=not self.verbose, miniters=100):
            test_pos_items = pos_items(gt_mat.getrow(user_idx))
            if len(test_pos_items) == 0:
                continue

            u_gt_pos = np.zeros(self.test_set.num_items, dtype=np.int)
            u_gt_pos[test_pos_items] = 1

            val_pos_items = [] if val_mat is None else pos_items(val_mat.getrow(user_idx))
            train_pos_items = [] if self.train_set.is_unk_user(user_idx) else pos_items(train_mat.getrow(user_idx))

            u_gt_neg = np.ones(self.test_set.num_items, dtype=np.int)
            u_gt_neg[test_pos_items + val_pos_items + train_pos_items] = 0

            item_indices = None if self.exclude_unknowns else np.arange(self.test_set.num_items)
            item_rank, item_scores = model.rank(user_idx, item_indices)

            for mt in metrics:
                mt_score = mt.compute(gt_pos=u_gt_pos, gt_neg=u_gt_neg,
                                      pd_rank=item_rank, pd_scores=item_scores)
                metric_user_results[mt.name][user_idx] = mt_score

        # avg results of ranking metrics
        for mt in metrics:
            metric_avg_results[mt.name] = sum(metric_user_results[mt.name].values()) / \
                                          len(metric_user_results[mt.name])

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

        ###########
        # FITTING #
        ###########
        if self.verbose:
            print('\n[{}] Training started!'.format(model.name))

        start = time.time()
        model.fit(self.train_set, self.val_set)
        train_time = time.time() - start

        ##############
        # EVALUATION #
        ##############
        if self.verbose:
            print('\n[{}] Evaluation started!'.format(model.name))

        metric_avg_results = OrderedDict()
        metric_user_results = defaultdict(defaultdict)

        start = time.time()
        self._rating_eval(model, rating_metrics, metric_avg_results, metric_user_results, user_based)
        self._ranking_eval(model, ranking_metrics, metric_avg_results, metric_user_results)
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
