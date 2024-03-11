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

from collections import OrderedDict

import numpy as np
from tqdm.auto import tqdm

from . import RatioSplit
from ..data import BasketDataset
from ..experiment.result import Result
from ..utils.common import safe_indexing


def ranking_eval(
    model,
    metrics,
    train_set,
    test_set,
    repetition_eval=False,
    exploration_eval=False,
    exclude_unknowns=True,
    verbose=False,
):
    """Evaluate model on provided ranking metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.BasketRecommender`, required
        BasketRecommender model to be evaluated.

    metrics: :obj:`iterable`, required
        List of rating metrics :obj:`cornac.metrics.RankingMetric`.

    train_set: :obj:`cornac.data.BasketDataset`, required
        BasketDataset to be used for model training. This will be used to exclude
        observations already appeared during training.

    test_set: :obj:`cornac.data.BasketDataset`, required
        BasketDataset to be used for evaluation.

    repetition_eval: boolean, optional,
        Evaluation on repetition items, appeared in history baskets.

    exploration_eval: boolean, optional,
        Evaluation on exploration items, not appeared in history baskets.

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

    avg_results = {
        "conventional": [],
        "repetition": [],
        "exploration": [],
    }
    user_results = {
        "conventional": [{} for _ in enumerate(metrics)],
        "repetition": [{} for _ in enumerate(metrics)],
        "exploration": [{} for _ in enumerate(metrics)],
    }

    def pos_items(baskets):
        return [item_idx for basket in baskets for item_idx in basket]

    def get_gt_items(train_set, test_set, test_pos_items, exclude_unknowns):
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

        u_gt_pos_items = np.nonzero(u_gt_pos_mask)[0]
        u_gt_neg_items = np.nonzero(u_gt_neg_mask)[0]
        item_indices = np.nonzero(u_gt_pos_mask + u_gt_neg_mask)[0]
        return item_indices, u_gt_pos_items, u_gt_neg_items

    (test_user_indices, *_) = test_set.uir_tuple
    for [user_idx], [bids], [(*history_baskets, gt_basket)] in tqdm(
        test_set.ubi_iter(batch_size=1, shuffle=False),
        total=len(set(test_user_indices)),
        desc="Ranking",
        disable=not verbose,
        miniters=100,
    ):
        test_pos_items = pos_items([gt_basket])
        if len(test_pos_items) == 0:
            continue

        item_indices, u_gt_pos_items, u_gt_neg_items = get_gt_items(
            train_set, test_set, test_pos_items, exclude_unknowns
        )

        item_rank, item_scores = model.rank(
            user_idx,
            item_indices,
            history_baskets=history_baskets,
            history_bids=bids[:-1],
            uir_tuple=test_set.uir_tuple,
            baskets=test_set.baskets,
            basket_indices=test_set.basket_indices,
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
            user_results["conventional"][i][user_idx] = mt_score

        history_items = set(
            item_idx for basket in history_baskets for item_idx in basket
        )
        if repetition_eval:
            test_repetition_pos_items = pos_items(
                [[iid for iid in gt_basket if iid in history_items]]
            )
            if len(test_repetition_pos_items) > 0:
                _, u_gt_pos_items, u_gt_neg_items = get_gt_items(
                    train_set, test_set, test_repetition_pos_items, exclude_unknowns
                )
                for i, mt in enumerate(metrics):
                    mt_score = mt.compute(
                        gt_pos=u_gt_pos_items,
                        gt_neg=u_gt_neg_items,
                        pd_rank=item_rank,
                        pd_scores=item_scores,
                        item_indices=item_indices,
                    )
                    user_results["repetition"][i][user_idx] = mt_score

        if exploration_eval:
            test_exploration_pos_items = pos_items(
                [[iid for iid in gt_basket if iid not in history_items]]
            )
            if len(test_exploration_pos_items) > 0:
                _, u_gt_pos_items, u_gt_neg_items = get_gt_items(
                    train_set, test_set, test_exploration_pos_items, exclude_unknowns
                )
                for i, mt in enumerate(metrics):
                    mt_score = mt.compute(
                        gt_pos=u_gt_pos_items,
                        gt_neg=u_gt_neg_items,
                        pd_rank=item_rank,
                        pd_scores=item_scores,
                        item_indices=item_indices,
                    )
                    user_results["exploration"][i][user_idx] = mt_score
    # avg results of ranking metrics
    for i, mt in enumerate(metrics):
        avg_results["conventional"].append(
            np.mean(list(user_results["conventional"][i].values()))
            if len(user_results["conventional"][i]) > 0
            else 0
        )
        if repetition_eval:
            avg_results["repetition"].append(
                np.mean(list(user_results["repetition"][i].values()))
                if len(user_results["repetition"][i]) > 0
                else 0
            )
        if exploration_eval:
            avg_results["exploration"].append(
                np.mean(list(user_results["exploration"][i].values()))
                if len(user_results["repetition"][i]) > 0
                else 0
            )

    return avg_results, user_results


class NextBasketEvaluation(RatioSplit):
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

    fmt: str, default: 'UBI'
        Format of the input data. Currently, we are supporting:

        'UBI': User, Basket, Item
        'UBIT': User, Basket, Item, Timestamp
        'UBITJson': User, Basket, Item, Timestamp, Json

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
        repetition_eval=False,
        exploration_eval=False,
        exclude_unknowns=True,
        verbose=False,
        **kwargs
    ):
        assert fmt.startswith("U")
        data_size = len(set(u for (u, *_) in data))  # number of users

        super().__init__(
            data=data,
            data_size=data_size,
            test_size=test_size,
            val_size=val_size,
            fmt=fmt,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            **kwargs
        )
        self.repetition_eval = repetition_eval
        self.exploration_eval = exploration_eval

    def _split(self):
        user_arr = [u for (u, *_) in self.data]
        all_users = np.unique(user_arr)
        self.rng.shuffle(all_users)

        train_users = set(all_users[: self.train_size])
        test_users = set(all_users[-self.test_size :])
        val_users = set(all_users[self.train_size : -self.test_size])

        train_idx = [i for i, u in enumerate(user_arr) if u in train_users]
        test_idx = [i for i, u in enumerate(user_arr) if u in test_users]
        val_idx = [i for i, u in enumerate(user_arr) if u in val_users]

        train_data = safe_indexing(self.data, train_idx)
        test_data = safe_indexing(self.data, test_idx)
        val_data = safe_indexing(self.data, val_idx) if len(val_idx) > 0 else None

        self.build(train_data=train_data, test_data=test_data, val_data=val_data)

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

    def eval(self, model, test_set, ranking_metrics, **kwargs):
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()

        avg_results, user_results = ranking_eval(
            model=model,
            metrics=ranking_metrics,
            train_set=self.train_set,
            test_set=test_set,
            repetition_eval=self.repetition_eval,
            exploration_eval=self.exploration_eval,
            exclude_unknowns=self.exclude_unknowns,
            verbose=self.verbose,
        )

        for i, mt in enumerate(ranking_metrics):
            metric_avg_results[mt.name] = avg_results["conventional"][i]
            metric_user_results[mt.name] = user_results["conventional"][i]

        if self.repetition_eval:
            for i, mt in enumerate(ranking_metrics):
                metric_avg_results["{}-rep".format(mt.name)] = avg_results[
                    "repetition"
                ][i]
                metric_user_results["{}-rep".format(mt.name)] = user_results[
                    "repetition"
                ][i]

        if self.repetition_eval:
            for i, mt in enumerate(ranking_metrics):
                metric_avg_results["{}-expl".format(mt.name)] = avg_results[
                    "exploration"
                ][i]
                metric_user_results["{}-expl".format(mt.name)] = user_results[
                    "exploration"
                ][i]
        return Result(model.name, metric_avg_results, metric_user_results)
