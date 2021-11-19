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

class PredefinedSplit(BaseMethod):
    """ Take a predefined training/testing split of your data set.
    TODO: Add validation sets.

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value)].

    train_data: array-like, required
        Raw preference training data in the triplet format [(user_id, item_id, rating_value)].

    test_data: array-like, required
        Raw preference testing data in the triplet format [(user_id, item_id, rating_value)].

    val_data: array-like, optional, default: None
        Raw preference validation data in the triplet format [(user_id, item_id, rating_value)].

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

    def __init__(self, data, train_data, test_data, val_data = None, rating_threshold=1.0, seed=None, exclude_unknowns=True, verbose=False, **kwargs):
        super().__init__(data=data, rating_threshold=rating_threshold, seed=seed,
                         exclude_unknowns=exclude_unknowns, verbose=verbose, **kwargs)


        self._build(train_data, test_data, val_data)

    def _build(self, train_data, test_data, val_data):
        self.build(train_data=train_data, test_data=test_data, val_data=val_data)
