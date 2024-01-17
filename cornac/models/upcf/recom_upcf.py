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

import itertools

import numpy as np
from scipy.sparse import csr_matrix, vstack

from ..recommender import NextBasketRecommender


class UPCF(NextBasketRecommender):
    """User Popularity-based CF (UPCF)

    Parameters
    ----------
    name: string, default: 'UPCF'
        The name of the recommender model.

    recency: int, optional, default: 1
        The size of recency window.
        If 0, all baskets will be used.

    locality: int, optional, default: 1
        The strength we enforce the similarity between two items within a basket

    asymmetry: float, optional, default: 0.25
        Trade-off parameter which balances the importance of the probability of having item i given j and probability having item j given i.
        This value will be computed via `similaripy.asymetric_cosine`.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    References
    ----------
    Guglielmo Faggioli, Mirko Polato, and Fabio Aiolli. 2020.
    Recency Aware Collaborative Filtering for Next Basket Recommendation.
    In Proceedings of the 28th ACM Conference on User Modeling, Adaptation and Personalization (UMAP '20). Association for Computing Machinery, New York, NY, USA, 80–87. https://doi.org/10.1145/3340631.3394850

    """

    def __init__(
        self,
        name="UPCF",
        recency=1,
        locality=1,
        asymmetry=0.25,
        verbose=False,
    ):
        super().__init__(name=name, trainable=False, verbose=verbose)
        self.recency = recency
        self.locality = locality
        self.asymmetry = asymmetry

    def fit(self, train_set, val_set=None):
        super().fit(train_set=train_set, val_set=val_set)
        self.user_wise_popularity = vstack(
            [
                self._get_user_wise_popularity(basket_items)
                for _, _, [basket_items] in train_set.ubi_iter(
                    batch_size=1, shuffle=False
                )
            ]
        )
        (u_indices, i_indices, r_values) = train_set.uir_tuple
        self.user_item_matrix = csr_matrix(
            (r_values, (u_indices, i_indices)),
            shape=(train_set.num_users, self.total_items),
            dtype="float32",
        )
        return self

    def _get_user_wise_popularity(self, basket_items):
        users = []
        items = []
        scores = []
        recent_basket_items = (
            basket_items[-self.recency :] if self.recency > 0 else basket_items
        )
        for iid in list(set(itertools.chain.from_iterable(recent_basket_items))):
            users.append(0)
            items.append(iid)
            denominator = (
                min(self.recency, len(recent_basket_items))
                if self.recency > 0
                else len(recent_basket_items)
            )
            numerator = sum([1 for items in recent_basket_items if iid in items])
            scores.append(numerator / denominator)
        return csr_matrix(
            (scores, (users, items)), shape=(1, self.total_items), dtype="float32"
        )

    def score(self, user_idx, history_baskets, **kwargs):
        import similaripy as sim

        items = list(set(itertools.chain.from_iterable(history_baskets)))
        current_user_item_matrix = csr_matrix(
            (np.ones(len(items)), (np.zeros(len(items)), items)),
            shape=(1, self.total_items),
            dtype="float32",
        )
        current_user_wise_popularity = self._get_user_wise_popularity(history_baskets)
        user_wise_popularity = vstack(
            [current_user_wise_popularity, self.user_wise_popularity]
        )
        user_item_matrix = vstack([current_user_item_matrix, self.user_item_matrix])
        user_sim = sim.asymmetric_cosine(
            user_item_matrix, alpha=self.asymmetry, target_rows=[0], verbose=False
        )
        scores = (
            sim.dot_product(
                user_sim.power(self.locality).tocsr()[0],
                user_wise_popularity,
                verbose=False,
            )
            .toarray()
            .squeeze()
        )

        return scores
