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

from collections import Counter

import numpy as np

from ..recommender import SequentialRecommender


class SPop(SequentialRecommender):
    """Recommend most popular items of the current session.

    Parameters
    ----------
    name: string, default: 'SPop'
        The name of the recommender model.

    mode: str, optional, default: 'session-based'
        One of 'session-based' or 'session-aware'. SPop's popularity logic
        is the same in both; the only difference is how
        :meth:`_flatten_history` collapses the nested ``history_items``
        coming from :class:`cornac.eval_methods.SequentialEvaluation`.

    use_session_popularity: boolean, optional, default: True
        When False, no item frequency from history items in current session
        are used; only the global training popularity is returned.

    References
    ----------
    Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk:
    Session-based Recommendations with Recurrent Neural Networks, ICLR 2016
    """

    def __init__(self, name="SPop", mode="session-based", use_session_popularity=True):
        super().__init__(name=name, mode=mode, trainable=False)
        self.use_session_popularity = use_session_popularity
        self.item_freq = Counter()

    def fit(self, train_set, val_set=None):
        super().fit(train_set=train_set, val_set=val_set)
        self.item_freq = Counter(self.train_set.uir_tuple[1])
        return self

    def score(self, user_idx, history_items, **kwargs):
        item_scores = np.zeros(self.total_items, dtype=np.float32)
        max_item_freq = max(self.item_freq.values()) if len(self.item_freq) > 0 else 1
        for iid, freq in self.item_freq.items():
            item_scores[iid] = freq / max_item_freq
        if self.use_session_popularity:
            flat_history = self._flatten_history(history_items)
            s_item_freq = Counter(flat_history)
            for iid, cnt in s_item_freq.most_common():
                item_scores[iid] += cnt
        return item_scores
