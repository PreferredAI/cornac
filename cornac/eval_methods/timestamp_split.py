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
from ..utils.common import safe_indexing


class TimestampSplit(BaseMethod):
    """Splitting data into training, validation, and test sets by absolute timestamp cutoffs.

    Given two timestamps `val_timestamp` and `test_timestamp`, interactions are partitioned as:

        train: timestamp < val_timestamp
        validation: val_timestamp <= timestamp < test_timestamp
        test: timestamp >= test_timestamp

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the quadruplet format [(user_id, item_id, rating_value, timestamp)].

    val_timestamp: int or float, required
        Cutoff between training and validation sets. Interactions with timestamp strictly
        less than this value go into the training set.

    test_timestamp: int or float, required
        Cutoff between validation and test sets. Interactions with timestamp greater than
        or equal to this value go into the test set. Must be greater than `val_timestamp`.

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
        val_timestamp,
        test_timestamp,
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

        if val_timestamp is None or test_timestamp is None:
            raise ValueError(
                "Both val_timestamp and test_timestamp are required."
            )

        if val_timestamp >= test_timestamp:
            raise ValueError(
                "val_timestamp ({}) must be strictly less than test_timestamp ({}).".format(
                    val_timestamp, test_timestamp
                )
            )

        self.val_timestamp = val_timestamp
        self.test_timestamp = test_timestamp

        self._split()

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
