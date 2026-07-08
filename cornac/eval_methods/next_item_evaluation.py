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
from ..utils import validate_format
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

    Notes
    -----
    **Data splitting.** Ratio-based splitting (inherited from
    :obj:`BaseMethod`) and per-user leave-last-out both leak future
    information into training: a random split trains on interactions that
    postdate the test items, while leave-last-out places each user's held-out
    item at a different absolute time, so training on one user includes other
    users' later interactions (popularity drift then distorts results). The
    recommended protocol is a single global temporal cutoff; use
    :meth:`from_timestamps` for a leakage-free, session-level split.
    Per-user leave-last-out remains available via :meth:`leave_last_out`
    for comparability with published results.

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
        base_model = getattr(model, "model", None)
        if not isinstance(model, NextItemRecommender) and not isinstance(
            base_model, NextItemRecommender
        ):
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

    @classmethod
    def from_timestamps(
        cls,
        data,
        test_timestamp,
        val_timestamp=None,
        fmt="USIT",
        exclude_unknowns=True,
        mode="last",
        seed=None,
        verbose=False,
        **kwargs,
    ):
        """Constructing evaluation method by a global temporal split.

        Sessions are split by a single global time horizon so that no
        training session is evaluated against future information shared across
        users. This is the leakage-free protocol recommended by the
        offline-evaluation literature (see Notes).

        Parameters
        ----------
        data: list, required
            Raw preference data in the tuple format given by `fmt`.

        test_timestamp: float, required
            Absolute timestamp (same unit as the timestamps in `data`) marking
            the start of the test period. Sessions whose last event is at or
            after this timestamp form the test set.

        val_timestamp: float, optional, default: None
            Absolute timestamp marking the start of the validation period. If
            None, no validation set is created and the training set absorbs
            everything before `test_timestamp`. Must be strictly smaller than
            `test_timestamp`.

        fmt: str, default: 'USIT'
            Format of the input data. Currently, we are supporting:

            'SIT': Session, Item, Timestamp
            'USIT': User, Session, Item, Timestamp
            'SITJson': Session, Item, Timestamp, Json
            'USITJson': User, Session, Item, Timestamp, Json

        exclude_unknowns: bool, optional, default: True
            Whether to exclude unknown users/items in evaluation.

        mode: str, optional, default: 'last'
            Evaluation mode is either 'next' or 'last'.
            If 'last', only evaluate the last item.
            If 'next', evaluate every next item in the sequence.

        seed: int, optional, default: None
            Random seed for reproducibility.

        verbose: bool, optional, default: False
            The verbosity flag.

        Returns
        -------
        method: :obj:`<cornac.eval_methods.NextItemEvaluation>`
            Evaluation method object.

        Notes
        -----
        Sessions are atomic: each session is assigned to exactly one split by
        the timestamp of its **last event** ``last_ts = max(t of session)``:

        - train: ``last_ts < val_timestamp``
        - val:   ``val_timestamp <= last_ts < test_timestamp``
        - test:  ``last_ts >= test_timestamp``

        ``>=`` goes to the later split (same convention as
        :obj:`cornac.eval_methods.TimestampSplit`). With
        ``val_timestamp=None``, train is ``last_ts < test_timestamp``.

        Assigning whole sessions by last event keeps sessions intact (an
        interaction-level cutoff would truncate straddling sessions), bounding
        residual leakage by session length (Hidasi & Czapp, RecSys 2023).
        Rows keep their original relative order within each partition, as
        required by :obj:`cornac.data.SequentialDataset.build`.

        References
        ----------
        Meng et al. (2020). Exploring Data Splitting Strategies for the
        Evaluation of Recommendation Models. RecSys 2020.

        Ji et al. (2023). A Critical Study on Data Leakage in Recommender
        System Offline Evaluation. ACM TOIS 2023.

        Hidasi & Czapp (2023). Widespread Flaws in Offline Evaluation of
        Recommender Systems. RecSys 2023.

        """
        fmt = validate_format(fmt, ["SIT", "USIT", "SITJson", "USITJson"])

        if val_timestamp is not None and val_timestamp >= test_timestamp:
            raise ValueError(
                "val_timestamp ({}) must be strictly smaller than "
                "test_timestamp ({}).".format(val_timestamp, test_timestamp)
            )

        sid_pos = 1 if fmt in ["USIT", "USITJson"] else 0
        ts_pos = 3 if fmt in ["USIT", "USITJson"] else 2

        # Pass 1: last event timestamp per session.
        last_ts = {}
        for tup in data:
            sid = tup[sid_pos]
            t = float(tup[ts_pos])
            if sid not in last_ts or t > last_ts[sid]:
                last_ts[sid] = t

        # Pass 2: route tuples, preserving original relative order.
        train_data, val_data, test_data = [], [], []
        for tup in data:
            ts = last_ts[tup[sid_pos]]
            if ts >= test_timestamp:
                test_data.append(tup)
            elif val_timestamp is not None and ts >= val_timestamp:
                val_data.append(tup)
            else:
                train_data.append(tup)

        if len(train_data) == 0:
            raise ValueError(
                "Empty train partition: no session ends before the cutoff "
                "({}).".format(test_timestamp if val_timestamp is None else val_timestamp)
            )
        if len(test_data) == 0:
            raise ValueError(
                "Empty test partition: no session ends at or after "
                "test_timestamp ({}).".format(test_timestamp)
            )
        if val_timestamp is not None and len(val_data) == 0:
            warnings.warn(
                "Empty validation partition: no session ends in "
                "[{}, {}). Proceeding with no validation set.".format(
                    val_timestamp, test_timestamp
                )
            )
            val_data = None

        if verbose:
            def _n_sessions(part):
                return len({tup[sid_pos] for tup in part}) if part else 0

            print("---")
            print("Global temporal split:")
            print(
                "Train: {} sessions, {} interactions".format(
                    _n_sessions(train_data), len(train_data)
                )
            )
            print(
                "Val: {} sessions, {} interactions".format(
                    _n_sessions(val_data), len(val_data) if val_data else 0
                )
            )
            print(
                "Test: {} sessions, {} interactions".format(
                    _n_sessions(test_data), len(test_data)
                )
            )

        return cls.from_splits(
            train_data=train_data,
            test_data=test_data,
            val_data=val_data,
            fmt=fmt,
            exclude_unknowns=exclude_unknowns,
            seed=seed,
            verbose=verbose,
            mode=mode,
            **kwargs,
        )

    @classmethod
    def leave_last_out(
        cls,
        data,
        fmt="UIRT",
        exclude_unknowns=True,
        mode="last",
        seed=None,
        verbose=False,
        **kwargs,
    ):
        """Constructing evaluation method by per-user leave-last-out.

        Each user's interactions are sorted chronologically and treated as one
        session (session id = user id). The last item is held out for testing
        and the second-to-last for validation — the common protocol in the
        sequential recommendation literature (SASRec, BERT4Rec, ...).

        Parameters
        ----------
        data: list, required
            Raw preference data in the quadruplet format
            [(user_id, item_id, rating, timestamp)]. Ratings are ignored
            (implicit next-item feedback).

        fmt: str, default: 'UIRT'
            Format of the input data. Only 'UIRT' is supported.

        exclude_unknowns: bool, optional, default: True
            Whether to exclude unknown users/items in evaluation.

        mode: str, optional, default: 'last'
            Evaluation mode is either 'next' or 'last'.
            If 'last', only evaluate the last item.
            If 'next', evaluate every next item in the sequence.

        seed: int, optional, default: None
            Random seed for reproducibility.

        verbose: bool, optional, default: False
            The verbosity flag.

        Returns
        -------
        method: :obj:`<cornac.eval_methods.NextItemEvaluation>`
            Evaluation method object.

        Notes
        -----
        Per user (chronological, stable sort — tied timestamps keep input
        order), with cumulative sequences:

        - train: ``seq[:-2]``
        - val:   ``seq[:-1]`` (target = second-to-last item)
        - test:  ``seq`` (target = last item)

        Users with fewer than 3 interactions are dropped from all splits.

        This protocol leaks future information across users: each held-out
        item sits at a different absolute time, so training includes other
        users' later interactions (Ji et al., A Critical Study on Data
        Leakage in Recommender System Offline Evaluation, ACM TOIS 2023).
        It is provided for comparability with published results; prefer
        :meth:`from_timestamps` for a leakage-free protocol.

        """
        fmt = validate_format(fmt, ["UIRT"])

        by_user = OrderedDict()
        for u, i, _, t in data:
            by_user.setdefault(u, []).append((float(t), i, t))

        train_data, val_data, test_data = [], [], []
        n_skipped = 0
        for u, events in by_user.items():
            if len(events) < 3:
                n_skipped += 1
                continue
            events.sort(key=lambda x: x[0])
            seq = [(u, u, i, t) for _, i, t in events]
            train_data.extend(seq[:-2])
            val_data.extend(seq[:-1])
            test_data.extend(seq)

        if len(train_data) == 0:
            raise ValueError(
                "Empty train set: no user has at least 3 interactions."
            )

        if verbose:
            print("---")
            print("Leave-last-out split (user = session):")
            print(
                "{} users kept, {} users with < 3 interactions dropped".format(
                    len(by_user) - n_skipped, n_skipped
                )
            )

        return cls.from_splits(
            train_data=train_data,
            test_data=test_data,
            val_data=val_data,
            fmt="USIT",
            exclude_unknowns=exclude_unknowns,
            seed=seed,
            verbose=verbose,
            mode=mode,
            **kwargs,
        )
