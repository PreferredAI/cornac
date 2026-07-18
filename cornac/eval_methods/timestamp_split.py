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

from .base_method import BaseMethod
from .ratio_split import RatioSplit
from ..utils.common import safe_indexing


class TimestampSplit(BaseMethod):
    """Splitting data into training, validation, and test sets chronologically by timestamp.

    The split point can be given in two mutually-exclusive ways:

    1. **Absolute cutoffs** — provide `val_timestamp` and `test_timestamp` directly.
    2. **Ratios** — provide `test_size` (and optionally `val_size`), and the cutoff
       timestamps are computed automatically so that (approximately) that proportion of
       interactions falls into each set.

    In both cases interactions are partitioned as:

        train: timestamp < val_timestamp
        validation: val_timestamp <= timestamp < test_timestamp
        test: timestamp >= test_timestamp

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the quadruplet format [(user_id, item_id, rating_value, timestamp)].

    val_timestamp: int or float, optional, default: None
        Cutoff between training and validation sets. Interactions with timestamp strictly
        less than this value go into the training set. Provide together with `test_timestamp`
        to split by absolute cutoffs; leave as `None` to split by ratio instead.

    test_timestamp: int or float, optional, default: None
        Cutoff between validation and test sets. Interactions with timestamp greater than
        or equal to this value go into the test set. Must be greater than `val_timestamp`.
        Provide together with `val_timestamp` to split by absolute cutoffs; leave as `None`
        to split by ratio instead.

    test_size: float, optional, default: None
        The proportion of the (chronologically latest) test set, counted by number of
        interactions. If > 1 it is treated as an absolute number of interactions. Used
        only when `val_timestamp`/`test_timestamp` are not given. Because the split keeps
        all interactions sharing a boundary timestamp on the same side (to avoid temporal
        leakage), the realized proportion is approximate when timestamps are tied.

    val_size: float, optional, default: None
        The proportion of the validation set (the interactions immediately preceding the
        test set), counted by number of interactions. If > 1 it is treated as an absolute
        number of interactions. Only used together with `test_size`.

    fmt: str, optional, default: 'UIRT'
        Format of the input data. Must be 'UIRT' since timestamps are required.

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
        val_timestamp=None,
        test_timestamp=None,
        test_size=None,
        val_size=None,
        fmt="UIRT",
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

        if fmt != "UIRT" or len(self.data[0]) != 4:
            raise ValueError(
                'Input data must be in "UIRT" format for splitting by timestamp.'
            )

        if val_timestamp is not None and test_timestamp is not None:
            # Absolute-cutoff mode.
            if val_timestamp >= test_timestamp:
                raise ValueError(
                    "val_timestamp ({}) must be strictly less than test_timestamp ({}).".format(
                        val_timestamp, test_timestamp
                    )
                )
            self.val_timestamp = val_timestamp
            self.test_timestamp = test_timestamp
        elif test_size is not None:
            # Ratio mode: derive cutoffs from the requested proportions.
            self.val_timestamp, self.test_timestamp = self._cutoffs_from_ratio(
                test_size=test_size, val_size=val_size
            )
        else:
            raise ValueError(
                "Provide either both val_timestamp and test_timestamp, or test_size "
                "(optionally with val_size) to split by ratio."
            )

        self._split()

    def _cutoffs_from_ratio(self, test_size, val_size):
        """Convert requested proportions into (val_timestamp, test_timestamp) cutoffs.

        Ratios are interpreted by interaction count: the chronologically latest
        ``test_size`` fraction of interactions forms the test set, and the fraction
        immediately before it forms the validation set. Returns cutoff timestamps to be
        consumed by :meth:`_split`; ties are handled there via `<`/`>=` thresholds.
        """
        data_size = len(self.data)
        train_count, val_count, test_count = RatioSplit.validate_size(
            val_size=val_size, test_size=test_size, data_size=data_size
        )

        if test_count == 0:
            raise ValueError(
                "test_size={} yields an empty test set.".format(test_size)
            )

        sorted_ts = sorted(row[3] for row in self.data)

        # Interactions from index (train_count + val_count) onward go to test.
        test_timestamp = sorted_ts[train_count + val_count]
        # Validation starts at index train_count; with no validation set the window is
        # empty (val_timestamp == test_timestamp).
        val_timestamp = sorted_ts[train_count] if val_count > 0 else test_timestamp

        return val_timestamp, test_timestamp

    def _split(self):
        train_idx = []
        val_idx = []
        test_idx = []

        for idx, row in enumerate(self.data):
            ts = row[3]
            if ts < self.val_timestamp:
                train_idx.append(idx)
            elif ts < self.test_timestamp:
                val_idx.append(idx)
            else:
                test_idx.append(idx)

        if len(train_idx) == 0:
            raise ValueError(
                "Training set is empty. val_timestamp may be too small."
            )
        if len(test_idx) == 0:
            raise ValueError(
                "Test set is empty. test_timestamp may be too large."
            )

        train_data = safe_indexing(self.data, train_idx)
        test_data = safe_indexing(self.data, test_idx)
        val_data = safe_indexing(self.data, val_idx) if len(val_idx) > 0 else None

        self.build(train_data=train_data, test_data=test_data, val_data=val_data)
