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

"""Evaluation of catalog bundle recommendation, completion and construction."""

import warnings
from collections import OrderedDict
from itertools import chain
from math import ceil

import numpy as np
from scipy.sparse import csr_matrix

from .base_method import BaseMethod
from .ratio_split import RatioSplit
from ..data import BundleConstructionDataset, BundleDataset, Dataset
from ..models.recommender import BundleConstructionRecommender
from ..utils import get_rng, validate_format


class BundleEvaluation(BaseMethod):
    """Bundle recommendation evaluation: rank existing catalog bundles for users.

    Use :meth:`from_splits` with user-bundle splits in UIR or UIRT format.

    Parameters
    ----------
    bundle_item_data: array-like, required
        (bundle, constituent item) pairs defining the known catalog. Every
        catalog bundle is a candidate, including bundles without training feedback.

    user_item_data: array-like, optional, default: None
        Independent training-time (user, item, rating) observations. Without
        ``user_ids``, only users seen in training user-bundle feedback are kept.

    user_ids, bundle_ids, constituent_item_ids: sequence, optional, default: None
        Ordered catalogs fixing numeric ID mappings, including unobserved
        entities. Explicit user IDs keep the item feedback of users without
        bundle feedback.

    test_mask: str, optional, default: 'train_and_validation'
        Positives excluded at test time: 'train' (CrossCBR reference protocol)
        or 'train_and_validation'. Model selection still uses validation data.

    **kwargs:
        Other :obj:`BaseMethod` options. ``exclude_unknowns`` applies to unseen
        users, since all catalog bundles are known.

    Notes
    -----
    Bundle and constituent item IDs are separate namespaces. Validation and test
    bundle feedback is never converted into user-item feedback; callers must
    supply training-time side observations only.
    """

    def __init__(
        self,
        bundle_item_data,
        user_item_data=None,
        user_ids=None,
        bundle_ids=None,
        constituent_item_ids=None,
        test_mask="train_and_validation",
        **kwargs,
    ):
        self.bundle_item_data = list(bundle_item_data)
        if not self.bundle_item_data:
            raise ValueError("bundle_item_data must define a nonempty catalog")
        self.user_item_data = None if user_item_data is None else list(user_item_data)
        self.user_ids = None if user_ids is None else tuple(user_ids)
        self.bundle_ids = None if bundle_ids is None else tuple(bundle_ids)
        self.constituent_item_ids = (
            None if constituent_item_ids is None else tuple(constituent_item_ids)
        )
        if test_mask not in ("train", "train_and_validation"):
            raise ValueError("test_mask must be train or train_and_validation")
        self.test_mask = test_mask
        super().__init__(**kwargs)

    def build(self, train_data, test_data, val_data=None):
        # Materialize iterators once before validation and dataset construction.
        train_data = None if train_data is None else list(train_data)
        test_data = None if test_data is None else list(test_data)
        val_data = None if val_data is None else list(val_data)
        self.val_set = None
        return super().build(train_data, test_data, val_data)

    @property
    def total_bundles(self):
        return self.total_items

    def _build_datasets(self, train_data, test_data, val_data=None):
        catalog = OrderedDict((bid, None) for bid, _ in self.bundle_item_data)
        if self.bundle_ids is not None:
            declared_bundles = BundleDataset.ordered_map(self.bundle_ids, "bundle_ids")
            if set(declared_bundles) != set(catalog):
                raise ValueError(
                    "declared bundle catalog must match membership bundles"
                )
            catalog = declared_bundles
        if self.user_ids is not None:
            self.global_uid_map.update(
                BundleDataset.ordered_map(self.user_ids, "user_ids")
            )
        item_map = (
            None
            if self.constituent_item_ids is None
            else BundleDataset.ordered_map(
                self.constituent_item_ids, "constituent_item_ids"
            )
        )
        for data in (train_data, test_data, val_data):
            if data is not None and any(row[1] not in catalog for row in data):
                raise ValueError(
                    "user-bundle observations contain bundles absent from the catalog"
                )
            if (
                data is not None
                and self.user_ids is not None
                and any(row[0] not in self.global_uid_map for row in data)
            ):
                raise ValueError("user-bundle observation contains an undeclared user")
        for bid in catalog:
            self.global_iid_map[bid] = len(self.global_iid_map)
        super()._build_datasets(train_data, test_data, val_data)
        side = self.user_item_data
        if side is not None:
            if self.user_ids is not None:
                if any(row[0] not in self.global_uid_map for row in side):
                    raise ValueError(
                        "user-item observation contains an undeclared user"
                    )
            else:
                train_users = set(self.train_set.uir_tuple[0])
                side = [
                    row
                    for row in side
                    if self.global_uid_map.get(row[0]) in train_users
                ]
        self.train_set = BundleDataset.from_dataset(
            self.train_set,
            self.bundle_item_data,
            side,
            item_map,
        )
        for name in ("test_set", "val_set"):
            dataset = getattr(self, name)
            if dataset is not None:
                setattr(
                    self,
                    name,
                    BundleDataset.from_dataset(
                        dataset,
                        self.bundle_item_data,
                        side,
                        self.train_set.constituent_iid_map,
                    ),
                )

    def eval(self, model, train_set, test_set, val_set, **kwargs):
        # Model selection still receives validation data; only test exclusions vary.
        excluded_validation = None if self.test_mask == "train" else val_set
        return super().eval(
            model=model,
            train_set=train_set,
            test_set=test_set,
            val_set=excluded_validation,
            **kwargs,
        )


def _group_bundles(data, fmt, owners):
    """Group bundle rows as bundle -> {item: timestamp}, recording single owners."""
    groups = OrderedDict()
    duplicates = 0
    for row in () if data is None else data:
        uid, bid, iid, *ts = row if fmt.startswith("U") else (None,) + tuple(row)
        if fmt.startswith("U") and owners.setdefault(bid, uid) != uid:
            raise ValueError("bundle {} has conflicting owners".format(bid))
        items = groups.setdefault(bid, OrderedDict())
        if iid in items:
            duplicates += 1
        else:
            items[iid] = int(ts[0]) if fmt.endswith("T") else None
    if duplicates > 0:
        warnings.warn("%d duplicated observations are removed!" % duplicates)
    return groups


def _csr(entries, shape):
    """CSR matrix from an {(row, column): value} mapping."""
    rows, cols = zip(*entries) if entries else ([], [])
    return csr_matrix((list(entries.values()), (rows, cols)), shape=shape)


class BundleConstructionEvaluation(BaseMethod):
    """Bundle completion and construction evaluation on held-out bundles.

    Completion gives some items of each held-out bundle and ranks its missing
    items; construction (``input_ratio=0``) gives none and conditions on a user.
    Queries are bundle instances on the Dataset user axis; targets are
    constituent items.

    Parameters
    ----------
    data: array-like, optional, default: None
        Bundle rows, split by bundle and never by row. Use :meth:`from_splits`
        for pre-split bundles.

    fmt: str, default: 'BI'
        Bundle row format, where U is the bundle's single owner:

        'BI': Bundle, Item
        'BIT': Bundle, Item, Timestamp (or position)
        'UBI': User, Bundle, Item
        'UBIT': User, Bundle, Item, Timestamp

    user_bundle_data: array-like, optional, default: None
        (user, bundle) pairs or UIR rows. When given, they replace owners as
        the users of held-out bundles.

    user_item_data: array-like, optional, default: None
        Training-time (user, item) pairs or UIR rows.

    test_size: float, optional, default: 0.2
        The proportion of bundles held out for test, \
        if > 1 then it is treated as the number of test bundles.

    val_size: float, optional, default: 0.0
        The proportion of bundles held out for validation, \
        if > 1 then it is treated as the number of validation bundles.

    input_ratio: float, optional, default: 0.5
        Share of each held-out bundle given as input: ceil(input_ratio * n)
        items, clamped to [1, n - 1]. 0 means construction from scratch.

    input_order: str, optional, default: 'random'
        'random' (seeded) or 'first' (stable timestamp order; BIT/UBIT only).

    min_bundle_size: int, optional, default: 2
        Held-out bundles with fewer items move to training.

    item_ids: sequence, optional, default: None
        Ordered item catalog. Without it, items are registered in sorted ID
        order, so that indices, which break score ties, never depend on the split.

    test_input_data, val_input_data: array-like, optional, default: None
        Given items of a published fixed split, as bundle rows; the test and
        validation rows are then the targets.

    set_metrics: bool, optional, default: True
        Also report ``Set-F1`` and ``Set-Jaccard`` between the targets T and
        |T| constructed items.

    exclude_unknowns, seed, verbose, **kwargs:
        As in :obj:`BaseMethod`, including modalities such as
        ``item_feature=FeatureModality(features=..., ids=...)``.

    Notes
    -----
    **Queries.** A held-out bundle with users yields one query
    ``(user_id, bundle_id)`` per user, all with the same given items; otherwise
    one query ``bundle_id``. With ``input_ratio=0``, held-out bundles without
    users are dropped with a warning.

    **Leakage.** Training rows hold full training bundles and given items only.
    Held-out bundles never reach a training row or a side relation:
    ``user_bundle_matrix`` links users to full training bundles only, and
    ``user_item_data`` must contain training-time observations only. Every
    catalog item is a candidate.

    **Metrics.** Ranking metrics use the unchanged ``ranking_eval``, which ranks
    the full catalog minus each query's given items, without negative sampling.
    Set metrics use ``model.construct`` for a
    :class:`cornac.models.BundleConstructionRecommender`, otherwise the top |T|
    ranked items: ``Set-F1`` = 2 tp / (|pred| + |T|) and ``Set-Jaccard`` =
    tp / (|pred| + |T| - tp). They use the true target size |T|; with |T|
    distinct items, Set-F1 equals set precision and recall.
    """

    def __init__(
        self,
        data=None,
        fmt="BI",
        user_bundle_data=None,
        user_item_data=None,
        test_size=0.2,
        val_size=0.0,
        input_ratio=0.5,
        input_order="random",
        min_bundle_size=2,
        item_ids=None,
        test_input_data=None,
        val_input_data=None,
        set_metrics=True,
        exclude_unknowns=True,
        seed=None,
        verbose=False,
        **kwargs,
    ):
        validate_format(fmt, ["BI", "BIT", "UBI", "UBIT"])
        if not 0 <= input_ratio <= 1:
            raise ValueError("input_ratio must be in [0, 1]")
        if input_order not in ("random", "first"):
            raise ValueError("input_order must be random or first")
        if input_order == "first" and not fmt.endswith("T"):
            raise ValueError("input_order='first' requires BIT or UBIT timestamps")
        if input_ratio > 0 and min_bundle_size < 2:
            raise ValueError("completion requires min_bundle_size >= 2")
        self.user_bundle_data = (
            None if user_bundle_data is None else list(user_bundle_data)
        )
        self.user_item_data = None if user_item_data is None else list(user_item_data)
        self.test_size = test_size
        self.val_size = val_size
        self.input_ratio = input_ratio
        self.input_order = input_order
        self.min_bundle_size = min_bundle_size
        self.item_ids = None if item_ids is None else tuple(item_ids)
        self.test_input_data = (
            None if test_input_data is None else list(test_input_data)
        )
        self.val_input_data = None if val_input_data is None else list(val_input_data)
        self.set_metrics = set_metrics
        super().__init__(
            data=None if data is None else list(data),
            fmt=fmt,
            exclude_unknowns=exclude_unknowns,
            seed=seed,
            verbose=verbose,
            **kwargs,
        )
        if data is not None:
            self._split()

    @classmethod
    def from_splits(cls, train_data, test_data, val_data=None, fmt="BI", **kwargs):
        """Build from bundle rows of disjoint bundle splits (full bundles).

        With ``test_input_data``/``val_input_data``, the test/val rows are the
        held-out targets and the inputs are given as-is. A bundle that appears in
        more than one split raises ValueError.
        """
        return cls(fmt=fmt, **kwargs).build(train_data, test_data, val_data)

    def _split(self):
        position = 1 if self.fmt.startswith("U") else 0
        bundles = OrderedDict()
        for row in self.data:
            bundles.setdefault(row[position], []).append(row)
        rows = list(bundles.values())
        train_size, val_size, _ = RatioSplit.validate_size(
            self.val_size, self.test_size, len(rows)
        )
        order = self.rng.permutation(len(rows))

        def pick(indices):
            return [row for i in sorted(indices) for row in rows[i]]

        self.build(
            pick(order[:train_size]),
            pick(order[train_size + val_size :]),
            pick(order[train_size : train_size + val_size]) or None,
        )

    def build(self, train_data, test_data, val_data=None):
        # Materialize iterators once before validation and dataset construction.
        train_data = None if train_data is None else list(train_data)
        test_data = None if test_data is None else list(test_data)
        val_data = None if val_data is None else list(val_data)
        return super().build(train_data, test_data, val_data)

    def _given_items(self, items, rng):
        """Split one held-out bundle's {item: timestamp} into given items and targets."""
        n = len(items)
        if self.input_ratio == 0:
            return OrderedDict(), items
        # Rounding keeps exact products such as 0.7 * 10 from taking an extra item.
        k = min(max(ceil(round(self.input_ratio * n, 6)), 1), n - 1)
        rows = list(items.items())
        if self.input_order == "first":
            order = sorted(range(n), key=lambda p: rows[p][1])
        else:
            order = rng.permutation(n)
        chosen = set(order[:k])
        return (
            OrderedDict(r for p, r in enumerate(rows) if p in chosen),
            OrderedDict(r for p, r in enumerate(rows) if p not in chosen),
        )

    def _register_items(self, groups):
        """Register the item catalog before any dataset is built."""
        declared = (
            None
            if self.item_ids is None
            else BundleDataset.ordered_map(self.item_ids, "item_ids")
        )
        observed = {
            iid for group in groups for items in group.values() for iid in items
        }
        if declared is not None and observed - declared.keys():
            raise ValueError("bundle rows contain an undeclared item")
        # Sorted IDs keep item indices, which break score ties, independent of the split.
        for iid in sorted(observed, key=str) if declared is None else declared:
            self.global_iid_map[iid] = len(self.global_iid_map)

    def _users(self, owners):
        """Map every user and list the users of each bundle."""
        owner_map = OrderedDict()
        for uid in chain(
            owners.values(),
            (row[0] for row in self.user_bundle_data or ()),
            (row[0] for row in self.user_item_data or ()),
        ):
            owner_map.setdefault(uid, len(owner_map))
        if self.user_bundle_data is None:
            return owner_map, {bid: [uid] for bid, uid in owners.items()}
        users = {}
        for uid, bid, *_ in self.user_bundle_data:
            users.setdefault(bid, {})[uid] = None
        return owner_map, users

    def _filter_held_out(self, train, held_out, users):
        """Move small held-out bundles to training and drop user-less construction queries."""
        dropped = 0
        for _, targets, inputs in held_out:
            for bid in list(targets):
                if len(targets[bid]) + len(inputs.get(bid, ())) < self.min_bundle_size:
                    train[bid] = {**inputs.pop(bid, {}), **targets.pop(bid)}
                elif self.input_ratio == 0 and bid not in inputs and not users.get(bid):
                    del targets[bid]
                    dropped += 1
        if dropped > 0:
            warnings.warn("%d held-out bundles without users are dropped!" % dropped)

    def _instances(self, train, held_out, users, owners):
        """Register training bundles, then test and validation queries, with their rows."""
        rng = get_rng(self.seed)
        rows = {"train": [], "test": [], "val": []}
        instance_owners = []
        for bid, items in train.items():
            self.global_uid_map[bid] = len(self.global_uid_map)
            instance_owners.append(owners.get(bid))
            rows["train"].extend((bid, iid, 1.0, ts) for iid, ts in items.items())
        for split, targets, inputs in held_out:
            for bid, items in targets.items():
                given, items = (
                    (inputs[bid], items)
                    if bid in inputs
                    else self._given_items(items, rng)
                )
                for uid in users.get(bid) or [None]:
                    key = bid if uid is None else (uid, bid)
                    self.global_uid_map[key] = len(self.global_uid_map)
                    instance_owners.append(uid)
                    rows["train"].extend(
                        (key, iid, 1.0, ts) for iid, ts in given.items()
                    )
                    rows[split].extend((key, iid, 1.0, ts) for iid, ts in items.items())
        return rows, instance_owners

    def _side_relations(self, owners, owner_map, train):
        """Links to full training bundles and training-time user-item feedback."""
        user_bundle = None
        if self.user_bundle_data is not None or owners:
            links = (
                ((row[0], row[1]) for row in self.user_bundle_data)
                if self.user_bundle_data is not None
                else ((uid, bid) for bid, uid in owners.items())
            )
            user_bundle = _csr(
                {
                    (owner_map[uid], self.global_uid_map[bid]): 1.0
                    for uid, bid in links
                    if bid in train
                },
                (len(owner_map), len(self.global_uid_map)),
            )
        user_item = None
        if self.user_item_data is not None:
            feedback = OrderedDict()
            for uid, iid, *rating in self.user_item_data:
                if iid in self.global_iid_map:
                    feedback.setdefault(
                        (owner_map[uid], self.global_iid_map[iid]),
                        float(rating[0]) if rating else 1.0,
                    )
            user_item = _csr(feedback, (len(owner_map), len(self.global_iid_map)))
        return user_bundle, user_item

    def _build_datasets(self, train_data, test_data, val_data=None):
        owners = OrderedDict()
        train, test, val, test_input, val_input = [
            _group_bundles(data, self.fmt, owners)
            for data in (
                train_data,
                test_data,
                val_data,
                self.test_input_data,
                self.val_input_data,
            )
        ]
        if (
            train.keys() & test.keys()
            or train.keys() & val.keys()
            or test.keys() & val.keys()
        ):
            raise ValueError("a bundle appears in more than one split")
        if test_input.keys() - test.keys() or val_input.keys() - val.keys():
            raise ValueError("input data contain bundles without held-out targets")
        self._register_items((train, test, val, test_input, val_input))
        owner_map, users = self._users(owners)
        held_out = (("test", test, test_input), ("val", val, val_input))
        self._filter_held_out(train, held_out, users)
        rows, instance_owners = self._instances(train, held_out, users, owners)
        owner_indices = np.array(
            [-1 if uid is None else owner_map[uid] for uid in instance_owners],
            dtype="int",
        )
        user_bundle, user_item = self._side_relations(owners, owner_map, train)

        def make(split, partial=None):
            dataset = Dataset.build(
                rows[split],
                fmt="UIRT" if self.fmt.endswith("T") else "UIR",
                global_uid_map=self.global_uid_map,
                global_iid_map=self.global_iid_map,
                seed=self.seed,
            )
            return BundleConstructionDataset(
                dataset, owner_map, owner_indices, user_bundle, user_item, partial
            )

        self.train_set = make("train", np.arange(len(train), self.total_users))
        self.test_set = make("test")
        self.val_set = make("val") if rows["val"] else None
        if self.verbose:
            print("---")
            print("Training bundles = {}".format(len(train)))
            print("Test instances = {}".format(len(set(self.test_set.uir_tuple[0]))))
            print("Total instances = {}".format(self.total_users))
            print("Total items = {}".format(self.total_items))

    def eval(self, model, train_set, test_set, val_set, **kwargs):
        result = super().eval(
            model=model,
            train_set=train_set,
            test_set=test_set,
            val_set=val_set,
            **kwargs,
        )
        if not self.set_metrics:
            return result
        f1, jaccard = {}, {}
        for idx in np.unique(test_set.uir_tuple[0]):
            given = train_set.bundle_items(idx)
            targets = set(test_set.bundle_items(idx).tolist())
            size = len(targets)
            if isinstance(model, BundleConstructionRecommender):
                pred = model.construct(idx, given, size)
            else:
                candidates = np.ones(train_set.num_items, dtype=bool)
                candidates[given] = False
                pred = model.rank(idx, np.flatnonzero(candidates), k=size)[0][:size]
            pred = set(np.asarray(pred).tolist())
            hits = len(pred & targets)
            f1[idx] = 2 * hits / (len(pred) + size)
            jaccard[idx] = hits / (len(pred) + size - hits)
        for name, values in (("Set-F1", f1), ("Set-Jaccard", jaccard)):
            result.metric_avg_results[name] = sum(values.values()) / len(values)
            result.metric_user_results[name] = values
        return result
