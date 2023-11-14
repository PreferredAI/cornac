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

from ..recommender import Recommender
from ..recommender import is_ann_supported


class BaseANN(Recommender):
    """Base class for a recommender model supporting Approximate Nearest Neighbor (ANN) search.

    Parameters
    ----------------
    model: object: :obj:`cornac.models.Recommender`, required
        Trained recommender model which to get user/item vectors from.

    name: str, required
        Name of the recommender model.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.
    """

    def __init__(self, model, name="BaseANN", verbose=False):
        super().__init__(name=name, verbose=verbose, trainable=False)

        if not is_ann_supported(model):
            raise ValueError(f"{model.name} doesn't support ANN search")

        # get basic attributes to be a proper recommender
        super().fit(train_set=model.train_set, val_set=model.val_set)

    def build_index(self):
        """Building index from the base recommender model.

        :raise NotImplementedError
        """
        raise NotImplementedError()

    def knn_query(self, query, k):
        """Implementing ANN search for a given query.

        Returns
        -------
        :raise NotImplementedError
        """
        raise NotImplementedError()

    def recommend(self, user_id, k=-1, remove_seen=False, train_set=None):
        """Generate top-K item recommendations for a given user. Backward compatibility.

        Parameters
        ----------
        user_id: str, required
            The original ID of user.

        k: int, optional, default=-1
            Cut-off length for recommendations, k=-1 will return ranked list of all items.

        remove_seen: bool, optional, default: False
            Remove seen/known items during training and validation from output recommendations.
            This might shrink the list of recommendations to be less than k.

        train_set: :obj:`cornac.data.Dataset`, optional, default: None
            Training dataset needs to be provided in order to remove seen items.

        Returns
        -------
        recommendations: list
            Recommended items in the form of their original IDs.
        """
        assert isinstance(user_id, str)
        return self.recommend_batch(
            batch_users=[user_id],
            k=k,
            remove_seen=remove_seen,
            train_set=train_set,
        )[0]

    def recommend_batch(self, batch_users, k=-1, remove_seen=False, train_set=None):
        """Generate top-K item recommendations for a given batch of users. This is to leverage
        parallelization provided by some ANN frameworks.

        Parameters
        ----------
        batch_users: list, required
            The original ID of users.

        k: int, optional, default=-1
            Cut-off length for recommendations, k=-1 will return ranked list of all items.

        remove_seen: bool, optional, default: False
            Remove seen/known items during training and validation from output recommendations.
            This might shrink the list of recommendations to be less than k.

        train_set: :obj:`cornac.data.Dataset`, optional, default: None
            Training dataset needs to be provided in order to remove seen items.

        Returns
        -------
        recommendations: list
            Recommended items in the form of their original IDs.
        """
        user_idx = [self.uid_map.get(uid, -1) for uid in batch_users]

        if any(i == -1 for i in user_idx):
            raise ValueError(f"{batch_users} is unknown to the model.")

        if k < -1 or k > self.total_items:
            raise ValueError(
                f"k={k} is invalid, there are {self.total_users} users in total."
            )

        query = self.user_vectors[user_idx]
        knn_items, distances = self.knn_query(query, k=k)

        if remove_seen:
            if train_set is None:
                raise ValueError("train_set must be provided to remove seen items.")
            filtered_knn_items = []
            for u, i in zip(user_idx, knn_items):
                if u >= train_set.csr_matrix.shape[0]:
                    continue
                seen_mask = np.in1d(
                    np.arange(i.size), train_set.csr_matrix.getrow(u).indices
                )
                filtered_knn_items.append(i[~seen_mask])
            knn_items = filtered_knn_items

        recommendations = [
            [self.item_ids[i] for i in knn_items[u]] for u in range(len(user_idx))
        ]
        return recommendations
