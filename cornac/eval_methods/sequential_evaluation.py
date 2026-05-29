# Copyright 2026 The Cornac Authors. All Rights Reserved.
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
from ..models import SequentialRecommender
from . import BaseMethod

EVALUATION_MODES = frozenset(["any", "first", "last"])


def ranking_eval(
    model,
    metrics,
    train_set,
    test_set,
    user_based=True,
    exclude_unknowns=True,
    mode="any",
    verbose=False,
):
    """Session-aware ranking evaluation.

    Iterates the ``test_set`` by user (``usi_iter``) so that a model sees a
    user's *prior* sessions as history when scoring the held-out session.
    The session list passed as ``history_items`` is nested
    (``list[list[int]]``), and is consumed by
    :meth:`cornac.models.SequentialRecommender._flatten_history` according
    to the model's ``mode``.

    Modes
    -----
    - ``"any"``:   all items in the last session are positives.
    - ``"first"``: only the first item of the last session is the positive.
    - ``"last"``:  only the last item of the last session is the positive;
                   the rest of that session is appended to ``history_items``.
    """
    if len(metrics) == 0:
        return [], []

    avg_results = []
    user_results = [defaultdict(list) for _ in enumerate(metrics)]

    user_sessions = defaultdict(list)
    for [user_idx], [sids], [mapped_ids], [session_items] in tqdm(
        test_set.usi_iter(batch_size=1, shuffle=False),
        total=test_set.num_users,
        desc="Ranking",
        disable=not verbose,
        miniters=100,
    ):
        if len(session_items) == 0 or len(session_items[-1]) == 0:
            continue
        user_sessions[user_idx].append(sids[-1])

        if mode == "any":
            test_pos_items = session_items[-1]
        elif mode == "first":
            test_pos_items = session_items[-1][0:1]
        else:  # "last"
            test_pos_items = session_items[-1][-1:]

        # ground-truth masks over the test_set item space
        u_gt_pos_mask = np.zeros(test_set.num_items, dtype="int")
        u_gt_pos_mask[test_pos_items] = 1

        u_gt_neg_mask = np.ones(test_set.num_items, dtype="int")
        u_gt_neg_mask[test_pos_items] = 0

        if exclude_unknowns:
            u_gt_pos_mask = u_gt_pos_mask[: train_set.num_items]
            u_gt_neg_mask = u_gt_neg_mask[: train_set.num_items]

        u_gt_pos_items = np.nonzero(u_gt_pos_mask)[0]
        u_gt_neg_items = np.nonzero(u_gt_neg_mask)[0]
        item_indices = np.nonzero(u_gt_pos_mask + u_gt_neg_mask)[0]

        if mode == "last":
            history_items = list(session_items[:-1]) + [session_items[-1][:-1]]
        else:
            history_items = list(session_items[:-1])

        item_rank, item_scores = model.rank(
            user_idx,
            item_indices,
            history_items=history_items,
            history_mapped_ids=mapped_ids[:-1],
            sessions=test_set.sessions,
            session_indices=test_set.session_indices,
            extra_data=test_set.extra_data,
            mode=mode,
        )

        for i, mt in enumerate(metrics):
            mt_score = mt.compute(
                gt_pos=u_gt_pos_items,
                gt_neg=u_gt_neg_items,
                pd_rank=item_rank,
                pd_scores=item_scores,
                item_indices=item_indices,
            )
            user_results[i][user_idx].append(mt_score)

    # average across users (user-based)
    for i in range(len(metrics)):
        user_ids = list(user_sessions.keys())
        if len(user_ids) == 0:
            avg_results.append(0.0)
            continue
        per_user = [np.mean(user_results[i][uid]) for uid in user_ids]
        avg_results.append(np.mean(per_user))

    return avg_results, user_results


class SequentialEvaluation(BaseMethod):
    """Next Session Recommendation Evaluation method.

    Iterates the test set by user and feeds the model the user's prior
    sessions as history when ranking items in the held-out last session.

    Parameters
    ----------
    data: list, optional
        Raw preference data in tuple format. Format defined by ``fmt``.

    test_size: float, default: 0.2
    val_size: float, default: 0.0
    fmt: str, default: 'USIT'
        One of 'SIT', 'USIT', 'SITJson', 'USITJson'. User information is
        required for ``mode="session-aware"`` on the model side; pure SIT
        formats only support session-based models.
    seed: int, optional
    mode: str, default: 'any'
        One of 'any', 'first', 'last'.
    exclude_unknowns: bool, default: True
    verbose: bool, default: False
    """

    def __init__(
        self,
        data=None,
        test_size=0.2,
        val_size=0.0,
        fmt="USIT",
        seed=None,
        mode="any",
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
                global_sid_map=self.global_sid_map,
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
        user_based=True,
        verbose=False,
        mode="any",
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
        if not isinstance(model, SequentialRecommender):
            raise ValueError("model must be a SequentialRecommender but '%s' is provided" % type(model))

        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()

        if self.verbose:
            print("\n[{}] Training started!".format(model.name))

        start = time.time()
        model.fit(self.train_set, self.val_set)
        train_time = time.time() - start

        if self.verbose:
            print("\n[{}] Evaluation started!".format(model.name))

        rating_metrics, ranking_metrics = self.organize_metrics(metrics)
        if len(rating_metrics) > 0:
            warnings.warn(
                "SequentialEvaluation only supports ranking metrics. "
                "The given rating metrics {} will be ignored!".format([mt.name for mt in rating_metrics])
            )

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
        fmt="USIT",
        exclude_unknowns=False,
        seed=None,
        verbose=False,
        **kwargs,
    ):
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
