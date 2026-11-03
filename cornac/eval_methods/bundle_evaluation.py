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

"""Evaluation of existing catalog bundle recommendation."""

from collections import OrderedDict

from .base_method import BaseMethod
from ..data import BundleDataset


class BundleEvaluation(BaseMethod):
    """Evaluate user-bundle feedback using ordinary Cornac ranking metrics.

    Use ``from_splits(train_data, test_data, bundle_item_data=..., ...)``.
    User-bundle splits use UIR/UIRT. Required ``bundle_item_data`` contains
    (bundle, constituent item) pairs defining the known catalog. Every catalog
    bundle is a candidate, including bundles without training feedback.

    Optional ``user_item_data`` contains independent training UIR observations;
    only users seen in training user-bundle feedback are retained. Callers must
    supply training-only side observations. Validation/test bundle feedback is
    never converted into user-item feedback. Bundle and constituent ID spaces
    are separate. ``exclude_unknowns`` applies to unseen users; all catalog
    bundles are known. Optional ordered ``user_ids``, ``bundle_ids`` and
    ``constituent_item_ids`` preserve declared numeric catalogs; explicit user IDs
    retain independent feedback for UI-only users. ``test_mask='train'`` excludes
    only training positives at test time, matching the CrossCBR reference; the
    default ``'train_and_validation'`` also excludes validation positives. Model
    selection still uses validation data. Other BaseMethod options apply unchanged.
    """

    def __init__(self, bundle_item_data, user_item_data=None, user_ids=None,
                 bundle_ids=None, constituent_item_ids=None,
                 test_mask="train_and_validation", **kwargs):
        self.bundle_item_data = list(bundle_item_data)
        if not self.bundle_item_data:
            raise ValueError("bundle_item_data must define a nonempty catalog")
        self.user_item_data = None if user_item_data is None else list(user_item_data)
        self.user_ids = None if user_ids is None else tuple(user_ids)
        self.bundle_ids = None if bundle_ids is None else tuple(bundle_ids)
        self.constituent_item_ids = (None if constituent_item_ids is None
                                     else tuple(constituent_item_ids))
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
                raise ValueError("declared bundle catalog must match membership bundles")
            catalog = declared_bundles
        if self.user_ids is not None:
            self.global_uid_map.update(BundleDataset.ordered_map(self.user_ids, "user_ids"))
        item_map = (None if self.constituent_item_ids is None else
                    BundleDataset.ordered_map(self.constituent_item_ids, "constituent_item_ids"))
        for data in (train_data, test_data, val_data):
            if data is not None and any(row[1] not in catalog for row in data):
                raise ValueError("user-bundle observations contain bundles absent from the catalog")
            if (data is not None and self.user_ids is not None
                    and any(row[0] not in self.global_uid_map for row in data)):
                raise ValueError("user-bundle observation contains an undeclared user")
        for bid in catalog:
            self.global_iid_map[bid] = len(self.global_iid_map)
        super()._build_datasets(train_data, test_data, val_data)
        side = self.user_item_data
        if side is not None:
            if self.user_ids is not None:
                if any(row[0] not in self.global_uid_map for row in side):
                    raise ValueError("user-item observation contains an undeclared user")
            else:
                train_users = set(self.train_set.uir_tuple[0])
                side = [row for row in side
                        if self.global_uid_map.get(row[0]) in train_users]
        self.train_set = BundleDataset.from_dataset(
            self.train_set, self.bundle_item_data, side, item_map,
        )
        for name in ("test_set", "val_set"):
            dataset = getattr(self, name)
            if dataset is not None:
                setattr(self, name, BundleDataset.from_dataset(
                    dataset, self.bundle_item_data, side,
                    self.train_set.constituent_iid_map,
                ))

    def eval(self, model, train_set, test_set, val_set, **kwargs):
        # Model selection still receives validation data; only test exclusions vary.
        excluded_validation = None if self.test_mask == "train" else val_set
        return super().eval(model=model, train_set=train_set, test_set=test_set,
                            val_set=excluded_validation, **kwargs)
