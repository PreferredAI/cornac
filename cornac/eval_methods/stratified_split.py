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

from math import ceil
from collections import defaultdict

from .base_method import BaseMethod
from .ratio_split import RatioSplit
from ..utils import get_rng
from ..utils.common import safe_indexing


class StratifiedSplit(BaseMethod):
    """Grouping data by user or item then splitting data into training, validation, and test sets.

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value, timestamp)].

    group_by: str, optional, default: 'user'
        Grouping by 'user' or 'item'.

    chrono: bool, optional, default False
        Data is ordered by reviewed time or not. If this option is True, data must be in 'UIRT' format.

    test_size: float, optional, default: 0.2
        The proportion of the test set, \
        if > 1 then it is treated as the size of the test set.

    val_size: float, optional, default: 0.0
        The proportion of the validation set, \
        if > 1 then it is treated as the size of the validation set.

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
        data,
        group_by="user",
        chrono=False,
        fmt="UIRT",
        test_size=0.2,
        val_size=0.0,
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs
    ):
        super().__init__(
            data=data,
            fmt=fmt,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            **kwargs
        )

        if group_by not in ["user", "item"]:
            raise ValueError(
                "group_by option must be either 'user' or 'item' but {}".format(
                    group_by
                )
            )

        if chrono and (fmt != "UIRT" or len(self._data[0]) != 4):
            raise ValueError(
                'Input data must be in "UIRT" format for sorting chronologically.'
            )

        self.chrono = chrono
        self.group_by = group_by
        self.val_size = val_size
        self.test_size = test_size

        self._split()

    def _split(self):
        data = (
            sorted(self._data, key=lambda x: x[3]) if self.chrono else self._data
        )  # sort data chronologically

        grouped_indices = defaultdict(list)
        for idx, (uid, iid, *_) in enumerate(data):
            if self.group_by == "user":
                grouped_indices[uid].append(idx)
            else:
                grouped_indices[iid].append(idx)

        train_idx = []
        test_idx = []
        val_idx = []

        for rating_indices in grouped_indices.values():
            n_ratings = len(rating_indices)
            n_train, _, n_test = RatioSplit.validate_size(
                self.val_size, self.test_size, n_ratings
            )

            if self.chrono:
                # training portion is chronologically sorted
                # validation and test portions are randomly selected
                rating_indices = (
                    rating_indices[:n_train]
                    + self.rng.permutation(rating_indices[n_train:]).tolist()
                )
            else:
                rating_indices = self.rng.permutation(rating_indices).tolist()

            train_idx += rating_indices[:n_train]
            test_idx += rating_indices[-n_test:]
            val_idx += rating_indices[n_train:-n_test]

        train_data = safe_indexing(data, train_idx)
        test_data = safe_indexing(data, test_idx)
        val_data = safe_indexing(data, val_idx) if len(val_idx) > 0 else None

        self.build(train_data=train_data, test_data=test_data, val_data=val_data)
