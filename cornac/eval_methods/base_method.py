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

from collections import OrderedDict
import time

import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

from ..data import FeatureModality
from ..data import TextModality, ReviewModality
from ..data import ImageModality
from ..data import GraphModality
from ..data import SentimentModality
from ..data import Dataset
from ..metrics import RatingMetric
from ..metrics import RankingMetric
from ..experiment.result import Result
from ..utils import get_rng


def rating_eval(model, metrics, test_set, user_based=False, verbose=False):
    """Evaluate model on provided rating metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.Recommender`, required
        Recommender model to be evaluated.

    metrics: :obj:`iterable`, required
        List of rating metrics :obj:`cornac.metrics.RatingMetric`.

    test_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for evaluation.

    user_based: bool, optional, default: False
        Evaluation mode. Whether results are averaging based on number of users or number of ratings.

    verbose: bool, optional, default: False
        Output evaluation progress.
        
    Returns
    -------
    res: (List, List)
        Tuple of two lists:
         - average result for each of the metrics
         - average result per user for each of the metrics

    """

    if len(metrics) == 0:
        return [], []

    avg_results = []
    user_results = []

    (u_indices, i_indices, r_values) = test_set.uir_tuple
    r_preds = np.fromiter(
        tqdm(
            (
                model.rate(user_idx, item_idx).item()
                for user_idx, item_idx in zip(u_indices, i_indices)
            ),
            desc="Rating",
            disable=not verbose,
            miniters=100,
            total=len(u_indices),
        ),
        dtype=np.float,
    )

    gt_mat = test_set.csr_matrix
    pd_mat = csr_matrix((r_preds, (u_indices, i_indices)), shape=gt_mat.shape)

    for mt in metrics:
        if user_based:  # averaging over users
            user_results.append(
                {
                    user_idx: mt.compute(
                        gt_ratings=gt_mat.getrow(user_idx).data,
                        pd_ratings=pd_mat.getrow(user_idx).data,
                    ).item()
                    for user_idx in test_set.user_indices
                }
            )
            avg_results.append(sum(user_results[-1].values()) / len(user_results[-1]))
        else:  # averaging over ratings
            user_results.append({})
            avg_results.append(mt.compute(gt_ratings=r_values, pd_ratings=r_preds))

    return avg_results, user_results


def ranking_eval(
    model,
    metrics,
    train_set,
    test_set,
    val_set=None,
    rating_threshold=1.0,
    exclude_unknowns=True,
    verbose=False,
):
    """Evaluate model on provided ranking metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.Recommender`, required
        Recommender model to be evaluated.

    metrics: :obj:`iterable`, required
        List of rating metrics :obj:`cornac.metrics.RankingMetric`.

    train_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for model training. This will be used to exclude
        observations already appeared during training.

    test_set: :obj:`cornac.data.Dataset`, required
        Dataset to be used for evaluation.

    val_set: :obj:`cornac.data.Dataset`, optional, default: None
        Dataset to be used for model selection. This will be used to exclude
        observations already appeared during validation.

    rating_threshold: float, optional, default: 1.0
        The threshold to convert ratings into positive or negative feedback.

    exclude_unknowns: bool, optional, default: True
        Ignore unknown users and items during evaluation.

    verbose: bool, optional, default: False
        Output evaluation progress.

    Returns
    -------
    res: (List, List)
        Tuple of two lists:
         - average result for each of the metrics
         - average result per user for each of the metrics

    """

    if len(metrics) == 0:
        return [], []

    avg_results = []
    user_results = [{} for _ in enumerate(metrics)]

    gt_mat = test_set.csr_matrix
    train_mat = train_set.csr_matrix
    val_mat = None if val_set is None else val_set.csr_matrix

    def pos_items(csr_row):
        return [
            item_idx
            for (item_idx, rating) in zip(csr_row.indices, csr_row.data)
            if rating >= rating_threshold
        ]

    for user_idx in tqdm(
        test_set.user_indices, desc="Ranking", disable=not verbose, miniters=100
    ):
        test_pos_items = pos_items(gt_mat.getrow(user_idx))
        if len(test_pos_items) == 0:
            continue

        u_gt_pos = np.zeros(test_set.num_items, dtype=np.int)
        u_gt_pos[test_pos_items] = 1

        val_pos_items = [] if val_mat is None else pos_items(val_mat.getrow(user_idx))
        train_pos_items = (
            []
            if train_set.is_unk_user(user_idx)
            else pos_items(train_mat.getrow(user_idx))
        )

        u_gt_neg = np.ones(test_set.num_items, dtype=np.int)
        u_gt_neg[test_pos_items + val_pos_items + train_pos_items] = 0

        item_indices = None if exclude_unknowns else np.arange(test_set.num_items)
        item_rank, item_scores = model.rank(user_idx, item_indices)

        for i, mt in enumerate(metrics):
            mt_score = mt.compute(
                gt_pos=u_gt_pos,
                gt_neg=u_gt_neg,
                pd_rank=item_rank,
                pd_scores=item_scores,
            )
            user_results[i][user_idx] = mt_score

    # avg results of ranking metrics
    for i, mt in enumerate(metrics):
        avg_results.append(sum(user_results[i].values()) / len(user_results[i]))

    return avg_results, user_results


class BaseMethod:
    """Base Evaluation Method

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value)].

    rating_threshold: float, optional, default: 1.0
        Threshold used to binarize rating values into positive or negative feedback for
        model evaluation using ranking metrics (rating metrics are not affected).

    seed: int, optional, default: None
        Random seed for reproducibility.

    exclude_unknowns: bool, optional, default: True
        If `True`, unknown users and items will be ignored during model evaluation.

    verbose: bool, optional, default: False
        Output running log.

    """

    def __init__(
        self,
        data=None,
        fmt="UIR",
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs
    ):
        self._data = data
        self.fmt = fmt
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.rating_threshold = rating_threshold
        self.exclude_unknowns = exclude_unknowns
        self.verbose = verbose
        self.seed = seed
        self.rng = get_rng(seed)
        self.global_uid_map = OrderedDict()
        self.global_iid_map = OrderedDict()

        self.user_feature = kwargs.get("user_feature", None)
        self.user_text = kwargs.get("user_text", None)
        self.user_image = kwargs.get("user_image", None)
        self.user_graph = kwargs.get("user_graph", None)
        self.item_feature = kwargs.get("item_feature", None)
        self.item_text = kwargs.get("item_text", None)
        self.item_image = kwargs.get("item_image", None)
        self.item_graph = kwargs.get("item_graph", None)
        self.sentiment = kwargs.get("sentiment", None)
        self.review_text = kwargs.get("review_text", None)

        if verbose:
            print("rating_threshold = {:.1f}".format(rating_threshold))
            print("exclude_unknowns = {}".format(exclude_unknowns))

    @property
    def total_users(self):
        return len(self.global_uid_map)

    @property
    def total_items(self):
        return len(self.global_iid_map)

    @property
    def user_feature(self):
        return self.__user_feature

    @property
    def user_text(self):
        return self.__user_text

    @user_feature.setter
    def user_feature(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, FeatureModality):
            raise ValueError(
                "input_modality has to be instance of FeatureModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_feature = input_modality

    @user_text.setter
    def user_text(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, TextModality):
            raise ValueError(
                "input_modality has to be instance of TextModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_text = input_modality

    @property
    def user_image(self):
        return self.__user_image

    @user_image.setter
    def user_image(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, ImageModality):
            raise ValueError(
                "input_modality has to be instance of ImageModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_image = input_modality

    @property
    def user_graph(self):
        return self.__user_graph

    @user_graph.setter
    def user_graph(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, GraphModality):
            raise ValueError(
                "input_modality has to be instance of GraphModality but {}".format(
                    type(input_modality)
                )
            )
        self.__user_graph = input_modality

    @property
    def item_feature(self):
        return self.__item_feature

    @property
    def item_text(self):
        return self.__item_text

    @item_feature.setter
    def item_feature(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, FeatureModality):
            raise ValueError(
                "input_modality has to be instance of FeatureModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_feature = input_modality

    @item_text.setter
    def item_text(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, TextModality):
            raise ValueError(
                "input_modality has to be instance of TextModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_text = input_modality

    @property
    def item_image(self):
        return self.__item_image

    @item_image.setter
    def item_image(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, ImageModality):
            raise ValueError(
                "input_modality has to be instance of ImageModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_image = input_modality

    @property
    def item_graph(self):
        return self.__item_graph

    @item_graph.setter
    def item_graph(self, input_modality):
        if input_modality is not None and not isinstance(input_modality, GraphModality):
            raise ValueError(
                "input_modality has to be instance of GraphModality but {}".format(
                    type(input_modality)
                )
            )
        self.__item_graph = input_modality

    @property
    def sentiment(self):
        return self.__sentiment

    @sentiment.setter
    def sentiment(self, input_modality):
        if input_modality is not None and not isinstance(
            input_modality, SentimentModality
        ):
            raise ValueError(
                "input_modality has to be instance of SentimentModality but {}".format(
                    type(input_modality)
                )
            )
        self.__sentiment = input_modality

    @property
    def review_text(self):
        return self.__review_text

    @review_text.setter
    def review_text(self, input_modality):
        if input_modality is not None and not isinstance(
            input_modality, ReviewModality
        ):
            raise ValueError(
                "input_modality has to be instance of ReviewModality but {}".format(
                    type(input_modality)
                )
            )
        self.__review_text = input_modality

    def _reset(self):
        """Reset the random number generator for reproducibility"""
        self.rng = get_rng(self.seed)
        self.test_set = self.test_set.reset()

    def _organize_metrics(self, metrics):
        """Organize metrics according to their types (rating or raking)

        Parameters
        ----------
        metrics: :obj:`iterable`
            List of metrics.

        """
        if isinstance(metrics, dict):
            self.rating_metrics = metrics.get("rating", [])
            self.ranking_metrics = metrics.get("ranking", [])
        elif isinstance(metrics, list):
            self.rating_metrics = []
            self.ranking_metrics = []
            for mt in metrics:
                if isinstance(mt, RatingMetric):
                    self.rating_metrics.append(mt)
                elif isinstance(mt, RankingMetric) and hasattr(mt.k, "__len__"):
                    self.ranking_metrics.extend(
                        [mt.__class__(k=_k) for _k in sorted(set(mt.k))]
                    )
                else:
                    self.ranking_metrics.append(mt)
        else:
            raise ValueError("Type of metrics has to be either dict or list!")

        # sort metrics by name
        self.rating_metrics = sorted(self.rating_metrics, key=lambda mt: mt.name)
        self.ranking_metrics = sorted(self.ranking_metrics, key=lambda mt: mt.name)

    def _build_datasets(self, train_data, test_data, val_data=None):
        self.train_set = Dataset.build(
            data=train_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            seed=self.seed,
            exclude_unknowns=False,
        )
        if self.verbose:
            print("---")
            print("Training data:")
            print("Number of users = {}".format(self.train_set.num_users))
            print("Number of items = {}".format(self.train_set.num_items))
            print("Number of ratings = {}".format(self.train_set.num_ratings))
            print("Max rating = {:.1f}".format(self.train_set.max_rating))
            print("Min rating = {:.1f}".format(self.train_set.min_rating))
            print("Global mean = {:.1f}".format(self.train_set.global_mean))

        self.test_set = Dataset.build(
            data=test_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            seed=self.seed,
            exclude_unknowns=self.exclude_unknowns,
        )
        if self.verbose:
            print("---")
            print("Test data:")
            print("Number of users = {}".format(len(self.test_set.uid_map)))
            print("Number of items = {}".format(len(self.test_set.iid_map)))
            print("Number of ratings = {}".format(self.test_set.num_ratings))
            print(
                "Number of unknown users = {}".format(
                    self.test_set.num_users - self.train_set.num_users
                )
            )
            print(
                "Number of unknown items = {}".format(
                    self.test_set.num_items - self.train_set.num_items
                )
            )

        if val_data is not None and len(val_data) > 0:
            self.val_set = Dataset.build(
                data=val_data,
                fmt=self.fmt,
                global_uid_map=self.global_uid_map,
                global_iid_map=self.global_iid_map,
                seed=self.seed,
                exclude_unknowns=self.exclude_unknowns,
            )
            if self.verbose:
                print("---")
                print("Validation data:")
                print("Number of users = {}".format(len(self.val_set.uid_map)))
                print("Number of items = {}".format(len(self.val_set.iid_map)))
                print("Number of ratings = {}".format(self.val_set.num_ratings))

        if self.verbose:
            print("---")
            print("Total users = {}".format(self.total_users))
            print("Total items = {}".format(self.total_items))

        self.train_set.total_users = self.total_users
        self.train_set.total_items = self.total_items

    def _build_modalities(self):
        for user_modality in [self.user_feature, self.user_text, self.user_image, self.user_graph]:
            if user_modality is None:
                continue
            user_modality.build(
                id_map=self.global_uid_map,
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for item_modality in [self.item_feature, self.item_text, self.item_image, self.item_graph]:
            if item_modality is None:
                continue
            item_modality.build(
                id_map=self.global_iid_map,
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for modality in [self.sentiment, self.review_text]:
            if modality is None:
                continue
            modality.build(
                uid_map=self.train_set.uid_map,
                iid_map=self.train_set.iid_map,
                dok_matrix=self.train_set.dok_matrix,
            )

        for data_set in [self.train_set, self.test_set, self.val_set]:
            if data_set is None:
                continue
            data_set.add_modalities(
                user_feature=self.user_feature,
                user_text=self.user_text,
                user_image=self.user_image,
                user_graph=self.user_graph,
                item_feature=self.item_feature,
                item_text=self.item_text,
                item_image=self.item_image,
                item_graph=self.item_graph,
                sentiment=self.sentiment,
                review_text=self.review_text,
            )

    def build(self, train_data, test_data, val_data=None):
        if train_data is None or len(train_data) == 0:
            raise ValueError("train_data is required but None or empty!")
        if test_data is None or len(test_data) == 0:
            raise ValueError("test_data is required but None or empty!")

        self.global_uid_map.clear()
        self.global_iid_map.clear()

        self._build_datasets(train_data, test_data, val_data)
        self._build_modalities()

        return self

    def _eval(self, model, test_set, val_set, user_based):
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()

        avg_results, user_results = rating_eval(
            model=model,
            metrics=self.rating_metrics,
            test_set=test_set,
            user_based=user_based,
            verbose=self.verbose,
        )
        for i, mt in enumerate(self.rating_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        avg_results, user_results = ranking_eval(
            model=model,
            metrics=self.ranking_metrics,
            train_set=self.train_set,
            test_set=test_set,
            val_set=val_set,
            rating_threshold=self.rating_threshold,
            exclude_unknowns=self.exclude_unknowns,
            verbose=self.verbose,
        )
        for i, mt in enumerate(self.ranking_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        return Result(model.name, metric_avg_results, metric_user_results)

    def evaluate(self, model, metrics, user_based, show_validation=True):
        """Evaluate given models according to given metrics

        Parameters
        ----------
        model: :obj:`cornac.models.Recommender`
            Recommender model to be evaluated.

        metrics: :obj:`iterable`
            List of metrics.

        user_based: bool, required 
            Evaluation strategy for the rating metrics. Whether results 
            are averaging based on number of users or number of ratings.

        show_validation: bool, optional, default: True 
            Whether to show the results on validation set (if exists).

        Returns
        -------
        res: :obj:`cornac.experiment.Result`
        """
        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()
        self._organize_metrics(metrics)

        ###########
        # FITTING #
        ###########
        if self.verbose:
            print("\n[{}] Training started!".format(model.name))

        start = time.time()
        model.fit(self.train_set, self.val_set)
        train_time = time.time() - start

        ##############
        # EVALUATION #
        ##############
        if self.verbose:
            print("\n[{}] Evaluation started!".format(model.name))

        start = time.time()
        test_result = self._eval(
            model=model,
            test_set=self.test_set,
            val_set=self.val_set,
            user_based=user_based,
        )
        test_time = time.time() - start
        test_result.metric_avg_results["Train (s)"] = train_time
        test_result.metric_avg_results["Test (s)"] = test_time

        val_result = None
        if show_validation and self.val_set is not None:
            start = time.time()
            val_result = self._eval(
                model=model, test_set=self.val_set, val_set=None, user_based=user_based
            )
            val_time = time.time() - start
            val_result.metric_avg_results["Time (s)"] = val_time

        return test_result, val_result

    @classmethod
    def from_splits(
        cls,
        train_data,
        test_data,
        val_data=None,
        fmt="UIR",
        rating_threshold=1.0,
        exclude_unknowns=False,
        seed=None,
        verbose=False,
        **kwargs
    ):
        """Constructing evaluation method given data.

        Parameters
        ----------
        train_data: array-like
            Training data

        test_data: array-like
            Test data

        val_data: array-like, optional, default: None
            Validation data

        fmt: str, default: 'UIR'
            Format of the input data. Currently, we are supporting:

            'UIR': User, Item, Rating
            'UIRT': User, Item, Rating, Timestamp

        rating_threshold: float, default: 1.0
            Threshold to decide positive or negative preferences.

        exclude_unknowns: bool, default: False
            Whether to exclude unknown users/items in evaluation.

        seed: int, optional, default: None
            Random seed for reproduce the splitting.

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        method: :obj:`<cornac.eval_methods.BaseMethod>`
            Evaluation method object.

        """
        method = cls(
            fmt=fmt,
            rating_threshold=rating_threshold,
            exclude_unknowns=exclude_unknowns,
            seed=seed,
            verbose=verbose,
            **kwargs
        )

        return method.build(
            train_data=train_data, test_data=test_data, val_data=val_data
        )
