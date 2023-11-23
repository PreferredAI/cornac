# Copyright 2023 The Cornac Authors. All Rights Reserved.
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

from math import ceil
from collections import OrderedDict

import numpy as np
from tqdm.auto import tqdm

from ..data import BasketDataset
from ..metrics import RankingMetric
from ..experiment.result import Result
from . import BaseMethod
from ..utils.common import safe_indexing
from .base_method import BaseMethod


def ranking_eval(
    model,
    metrics,
    train_set,
    test_set,
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

    def pos_items(baskets):
        return [item_idx for basket in baskets for item_idx in basket]

    test_user_indices = set(test_set.uir_tuple[0])
    for user_idx in tqdm(
        test_user_indices, desc="Ranking", disable=not verbose, miniters=100
    ):
        test_pos_items = pos_items(
            [
                [test_set.uir_tuple[1][idx] for idx in test_set.baskets[bid]]
                for bid in test_set.user_basket_data[user_idx][-1:]
            ]
        )
        if len(test_pos_items) == 0:
            continue

        # binary mask for ground-truth positive items
        u_gt_pos_mask = np.zeros(test_set.num_items, dtype="int")
        u_gt_pos_mask[test_pos_items] = 1

        # binary mask for ground-truth negative items, removing all positive items
        u_gt_neg_mask = np.ones(test_set.num_items, dtype="int")
        u_gt_neg_mask[test_pos_items] = 0

        # filter items being considered for evaluation
        if exclude_unknowns:
            u_gt_pos_mask = u_gt_pos_mask[: train_set.num_items]
            u_gt_neg_mask = u_gt_neg_mask[: train_set.num_items]

        item_indices = np.nonzero(u_gt_pos_mask + u_gt_neg_mask)[0]
        u_gt_pos_items = np.nonzero(u_gt_pos_mask)[0]
        u_gt_neg_items = np.nonzero(u_gt_neg_mask)[0]

        item_rank, item_scores = model.rank(
            user_idx,
            [
                [test_set.uir_tuple[1][idx] for idx in test_set.baskets[bid]]
                for bid in test_set.user_basket_data[user_idx][:-1]
            ],
            item_indices,
            baskets=test_set.baskets,
            basket_ids=test_set.basket_ids,
            extra_data=test_set.extra_data,
        )

        for i, mt in enumerate(metrics):
            mt_score = mt.compute(
                gt_pos=u_gt_pos_items,
                gt_neg=u_gt_neg_items,
                pd_rank=item_rank,
                pd_scores=item_scores,
                item_indices=item_indices,
            )
            user_results[i][user_idx] = mt_score

    # avg results of ranking metrics
    for i, mt in enumerate(metrics):
        avg_results.append(sum(user_results[i].values()) / len(user_results[i]))

    return avg_results, user_results


class NextBasketEvaluation(BaseMethod):
    """Next Basket Recommendation Evaluation method

    Parameters
    ----------
    data: list, required
        Raw preference data in the tuple format [(user_id, baskets)].

    test_size: float, optional, default: 0.2
        The proportion of the test set, \
        if > 1 then it is treated as the size of the test set.

    val_size: float, optional, default: 0.0
        The proportion of the validation set, \
        if > 1 then it is treated as the size of the validation set.

    seed: int, optional, default: None
        Random seed for reproducibility.

    exclude_unknowns: bool, optional, default: True
        If `True`, unknown items will be ignored during model evaluation.

    verbose: bool, optional, default: False
        Output running log.

    """

    def __init__(
        self,
        data=None,
        test_size=0.2,
        val_size=0.0,
        fmt="UBI",
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs
    ):
        super().__init__(
            data=data,
            fmt=fmt,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            kwargs=kwargs,
        )

        self.train_size, self.val_size, self.test_size = self.validate_size(
            val_size, test_size, len(self._data)
        )
        self._split()

        if verbose:
            print("exclude_unknowns = {}".format(exclude_unknowns))

    @staticmethod
    def validate_size(val_size, test_size, num_users):
        if val_size is None:
            val_size = 0.0
        elif val_size < 0:
            raise ValueError("val_size={} should be greater than zero".format(val_size))
        elif val_size >= num_users:
            raise ValueError(
                "val_size={} should be less than the number of users {}".format(
                    val_size, num_users
                )
            )

        if test_size is None:
            test_size = 0.0
        elif test_size < 0:
            raise ValueError(
                "test_size={} should be greater than zero".format(test_size)
            )
        elif test_size >= num_users:
            raise ValueError(
                "test_size={} should be less than the number of users {}".format(
                    test_size, num_users
                )
            )

        if val_size < 1:
            val_size = ceil(val_size * num_users)
        if test_size < 1:
            test_size = ceil(test_size * num_users)

        if val_size + test_size >= num_users:
            raise ValueError(
                "The sum of val_size and test_size ({}) should be smaller than the number of users {}".format(
                    val_size + test_size, num_users
                )
            )

        train_size = num_users - (val_size + test_size)

        return int(train_size), int(val_size), int(test_size)

    def _split(self):
        data_idx = self.rng.permutation(len(self._data))
        train_idx = data_idx[: self.train_size]
        test_idx = data_idx[-self.test_size :]
        val_idx = data_idx[self.train_size : -self.test_size]

        train_data = safe_indexing(self._data, train_idx)
        test_data = safe_indexing(self._data, test_idx)
        val_data = safe_indexing(self._data, val_idx) if len(val_idx) > 0 else None

        self.build(train_data=train_data, test_data=test_data, val_data=val_data)

    def _organize_metrics(self, metrics):
        """Organize metrics according to their types (rating or raking)

        Parameters
        ----------
        metrics: :obj:`iterable`
            List of metrics.

        """
        if isinstance(metrics, dict):
            self.ranking_metrics = metrics.get("ranking", [])
        elif isinstance(metrics, list):
            self.ranking_metrics = []
            for mt in metrics:
                if isinstance(mt, RankingMetric) and hasattr(mt.k, "__len__"):
                    self.ranking_metrics.extend(
                        [mt.__class__(k=_k) for _k in sorted(set(mt.k))]
                    )
                else:
                    self.ranking_metrics.append(mt)
        else:
            raise ValueError("Type of metrics has to be either dict or list!")

        # sort metrics by name
        self.ranking_metrics = sorted(self.ranking_metrics, key=lambda mt: mt.name)

    def _build_datasets(self, train_data, test_data, val_data=None):
        self.train_set = BasketDataset.build(
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
            print("Number of baskets = {}".format(self.train_set.num_baskets))

        self.test_set = BasketDataset.build(
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
            print("Number of baskets = {}".format(self.test_set.num_baskets))
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
            self.val_set = BasketDataset.build(
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
                print("Number of baskets = {}".format(self.val_set.num_baskets))

        self.total_baskets = 0 if self.val_set is None else self.val_set.num_baskets
        self.total_baskets += self.test_set.num_baskets + self.train_set.num_baskets
        if self.verbose:
            print("---")
            print("Total users = {}".format(self.total_users))
            print("Total items = {}".format(self.total_items))
            print("Total baskets = {}".format(self.total_baskets))

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

    def _eval(self, model, test_set, **kwargs):
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()

        avg_results, user_results = ranking_eval(
            model=model,
            metrics=self.ranking_metrics,
            train_set=self.train_set,
            test_set=test_set,
            exclude_unknowns=self.exclude_unknowns,
            verbose=self.verbose,
        )
        for i, mt in enumerate(self.ranking_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        return Result(model.name, metric_avg_results, metric_user_results)
