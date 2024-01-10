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

import time
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
from tqdm.auto import tqdm

from ..data import SequentialDataset
from ..experiment.result import Result
from ..models import NextItemRecommender
from . import BaseMethod

EVALUATION_MODES = frozenset([
    "last",
    "next",
])

def ranking_eval(
    model,
    metrics,
    train_set,
    test_set,
    user_based=False,
    exclude_unknowns=True,
    mode="last",
    verbose=False,
):
    """Evaluate model on provided ranking metrics.

    Parameters
    ----------
    model: :obj:`cornac.models.NextItemRecommender`, required
        NextItemRecommender model to be evaluated.

    metrics: :obj:`iterable`, required
        List of rating metrics :obj:`cornac.metrics.RankingMetric`.

    train_set: :obj:`cornac.data.SequentialDataset`, required
        SequentialDataset to be used for model training. This will be used to exclude
        observations already appeared during training.

    test_set: :obj:`cornac.data.SequentialDataset`, required
        SequentialDataset to be used for evaluation.

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
    session_results = [defaultdict(list) for _ in enumerate(metrics)]
    user_results = [defaultdict(list) for _ in enumerate(metrics)]

    user_sessions = defaultdict(list)
    session_ids = []
    for [sid], [mapped_ids], [session_items] in tqdm(
        test_set.si_iter(batch_size=1, shuffle=False),
        total=len(test_set.sessions),
        desc="Ranking",
        disable=not verbose,
        miniters=100,
    ):
        if len(session_items) < 2:  # exclude all session with size smaller than 2
            continue
        user_idx = test_set.uir_tuple[0][mapped_ids[0]]
        if user_based:
            user_sessions[user_idx].append(sid)
        session_ids.append(sid)

        start_pos = 1 if mode == "next" else len(session_items) - 1
        for test_pos in range(start_pos, len(session_items), 1):
            test_pos_items = session_items[test_pos]

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

            item_rank, item_scores = model.rank(
                user_idx,
                item_indices,
                history_items=session_items[:test_pos],
                history_mapped_ids=mapped_ids[:test_pos],
                sessions=test_set.sessions,
                session_indices=test_set.session_indices,
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
                if user_based:
                    user_results[i][user_idx].append(mt_score)
                else:
                    session_results[i][sid].append(mt_score)

    # avg results of ranking metrics
    for i, mt in enumerate(metrics):
        if user_based:
            user_ids = list(user_sessions.keys())
            user_avg_results = [np.mean(user_results[i][user_idx]) for user_idx in user_ids]
            avg_results.append(np.mean(user_avg_results))
        else:
            session_result = [score for sid in session_ids for score in session_results[i][sid]]
            avg_results.append(np.mean(session_result))
    return avg_results, user_results


class NextItemEvaluation(BaseMethod):
    """Next Item Recommendation Evaluation method

    Parameters
    ----------
    data: list, required
        Raw preference data in the tuple format [(user_id, sessions)].

    test_size: float, optional, default: 0.2
        The proportion of the test set, \
        if > 1 then it is treated as the size of the test set.

    val_size: float, optional, default: 0.0
        The proportion of the validation set, \
        if > 1 then it is treated as the size of the validation set.

    fmt: str, default: 'SIT'
        Format of the input data. Currently, we are supporting:

        'SIT': Session, Item, Timestamp
        'USIT': User, Session, Item, Timestamp
        'SITJson': Session, Item, Timestamp, Json
        'USITJson': User, Session, Item, Timestamp, Json

    seed: int, optional, default: None
        Random seed for reproducibility.

    mode: str, optional, default: 'last'
        Evaluation mode is either 'next' or 'last'.
        If 'last', only evaluate the last item.
        If 'next', evaluate every next item in the sequence,

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
        fmt="SIT",
        seed=None,
        mode="last",
        exclude_unknowns=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__(
            data=data,
            data_size=0 if data is None else len(data),
            test_size=test_size,
            val_size=val_size,
            fmt=fmt,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            mode=mode,
            **kwargs,
        )
        
        if mode not in EVALUATION_MODES:
            raise ValueError(f"{mode} is not supported. ({EVALUATION_MODES})")
            
        self.mode = mode
        self.global_sid_map = kwargs.get("global_sid_map", OrderedDict())

    def _build_datasets(self, train_data, test_data, val_data=None):
        self.train_set = SequentialDataset.build(
            data=train_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            global_sid_map=self.global_sid_map,
            seed=self.seed,
            exclude_unknowns=False,
        )
        if self.verbose:
            print("---")
            print("Training data:")
            print("Number of users = {}".format(self.train_set.num_users))
            print("Number of items = {}".format(self.train_set.num_items))
            print("Number of sessions = {}".format(self.train_set.num_sessions))

        self.test_set = SequentialDataset.build(
            data=test_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            global_sid_map=self.global_sid_map,
            seed=self.seed,
            exclude_unknowns=self.exclude_unknowns,
        )
        if self.verbose:
            print("---")
            print("Test data:")
            print("Number of users = {}".format(len(self.test_set.uid_map)))
            print("Number of items = {}".format(len(self.test_set.iid_map)))
            print("Number of sessions = {}".format(self.test_set.num_sessions))
            print("Number of unknown users = {}".format(self.test_set.num_users - self.train_set.num_users))
            print("Number of unknown items = {}".format(self.test_set.num_items - self.train_set.num_items))

        if val_data is not None and len(val_data) > 0:
            self.val_set = SequentialDataset.build(
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
                print("Number of sessions = {}".format(self.val_set.num_sessions))

        self.total_sessions = 0 if self.val_set is None else self.val_set.num_sessions
        self.total_sessions += self.test_set.num_sessions + self.train_set.num_sessions
        if self.verbose:
            print("---")
            print("Total users = {}".format(self.total_users))
            print("Total items = {}".format(self.total_items))
            print("Total sessions = {}".format(self.total_sessions))

    @staticmethod
    def eval(
        model,
        train_set,
        test_set,
        exclude_unknowns,
        ranking_metrics,
        user_based=False,
        verbose=False,
        mode="last",
        **kwargs,
    ):
        metric_avg_results = OrderedDict()
        metric_user_results = OrderedDict()

        avg_results, user_results = ranking_eval(
            model=model,
            metrics=ranking_metrics,
            train_set=train_set,
            test_set=test_set,
            user_based=user_based,
            exclude_unknowns=exclude_unknowns,
            mode=mode,
            verbose=verbose,
        )

        for i, mt in enumerate(ranking_metrics):
            metric_avg_results[mt.name] = avg_results[i]
            metric_user_results[mt.name] = user_results[i]

        return Result(model.name, metric_avg_results, metric_user_results)

    def evaluate(self, model, metrics, user_based, show_validation=True):
        """Evaluate given models according to given metrics. Supposed to be called by Experiment.

        Parameters
        ----------
        model: :obj:`cornac.models.NextItemRecommender`
            NextItemRecommender model to be evaluated.

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
        if not isinstance(model, NextItemRecommender):
            raise ValueError("model must be a NextItemRecommender but '%s' is provided" % type(model))

        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()

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

        rating_metrics, ranking_metrics = self.organize_metrics(metrics)
        if len(rating_metrics) > 0:
            warnings.warn("NextItemEvaluation only supports ranking metrics. The given rating metrics {} will be ignored!".format([mt.name for mt in rating_metrics]))

        start = time.time()
        model.transform(self.test_set)
        test_result = self.eval(
            model=model,
            train_set=self.train_set,
            test_set=self.test_set,
            val_set=self.val_set,
            exclude_unknowns=self.exclude_unknowns,
            ranking_metrics=ranking_metrics,
            user_based=user_based,
            mode=self.mode,
            verbose=self.verbose,
        )
        test_time = time.time() - start
        test_result.metric_avg_results["Train (s)"] = train_time
        test_result.metric_avg_results["Test (s)"] = test_time

        val_result = None
        if show_validation and self.val_set is not None:
            start = time.time()
            model.transform(self.val_set)
            val_result = self.eval(
                model=model,
                train_set=self.train_set,
                test_set=self.val_set,
                val_set=None,
                exclude_unknowns=self.exclude_unknowns,
                ranking_metrics=ranking_metrics,
                user_based=user_based,
                mode=self.mode,
                verbose=self.verbose,
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
        fmt="SIT",
        exclude_unknowns=False,
        seed=None,
        verbose=False,
        **kwargs,
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

        fmt: str, default: 'SIT'
            Format of the input data. Currently, we are supporting:

            'SIT': Session, Item, Timestamp
            'USIT': User, Session, Item, Timestamp
            'SITJson': Session, Item, Timestamp, Json
            'USITJson': User, Session, Item, Timestamp, Json

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
        method: :obj:`<cornac.eval_methods.NextItemEvaluation>`
            Evaluation method object.

        """
        method = cls(
            fmt=fmt,
            exclude_unknowns=exclude_unknowns,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )

        return method.build(
            train_data=train_data,
            test_data=test_data,
            val_data=val_data,
        )
