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

import numpy as np
from collections import Counter

from ..recommender import NextBasketRecommender


class GPTop(NextBasketRecommender):
    """Global Personalized Top Frequent Items.

    Parameters
    ----------
    name: string, default: 'GPTop'
        The name of the recommender model.

    use_global_popularity: boolean, optional, default: True
        When False, no item frequency from all users' baskets are being used.

    use_personalized_popularity: boolean, optional, default: True
        When False, no item frequency from history baskets are being used.

    References
    ----------
    Ming Li, Sami Jullien, Mozhdeh Ariannezhad, and Maarten de Rijke. 2023.
    A Next Basket Recommendation Reality Check.
    ACM Trans. Inf. Syst. 41, 4, Article 116 (October 2023), 29 pages. https://doi.org/10.1145/3587153

    """

    def __init__(
        self, name="GPTop", use_global_popularity=True, use_personalized_popularity=True
    ):
        super().__init__(name=name, trainable=False)
        self.use_global_popularity = use_global_popularity
        self.use_personalized_popularity = use_personalized_popularity
        self.item_freq = Counter()

    def fit(self, train_set, val_set=None):
        super().fit(train_set=train_set, val_set=val_set)
        if self.use_global_popularity:
            self.item_freq = Counter(self.train_set.uir_tuple[1])
        return self

    def score(self, user_idx, history_baskets, **kwargs):
        item_scores = np.ones(self.total_items)
        if self.use_global_popularity:
            for iid, freq in self.item_freq.items():
                item_scores[iid] = freq

        if self.use_personalized_popularity:
            p_item_freq = Counter([iid for iids in history_baskets for iid in iids])

            max_item_freq = (
                max(self.item_freq.values()) if len(self.item_freq) > 0 else 1
            )
            for iid, cnt in p_item_freq.most_common():
                item_scores[iid] = max_item_freq + cnt
        return item_scores
