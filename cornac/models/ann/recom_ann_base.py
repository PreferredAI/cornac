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


from ..recommender import Recommender
from ..recommender import is_ann_supported


class BaseANN(Recommender):
    """Base class for a recommender model supporting Approximate Nearest Neighbor (ANN) search.

    Parameters
    ----------------
    recom: object: :obj:`cornac.models.Recommender`, required
        Trained recommender model which to get user/item vectors from.

    name: str, required
        Name of the recommender model.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.
    """

    def __init__(self, recom, name="BaseANN", verbose=False):
        super().__init__(name=name, verbose=verbose, trainable=False)

        if not is_ann_supported(recom):
            raise ValueError(f"{recom.name} doesn't support ANN search")
        self.recom = recom

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
        """Generate top-K item recommendations for a given user.

        Parameters
        ----------
        user_id: str or list, required
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
        if isinstance(user_id, str):
            user_idx = [self.recom.uid_map.get(user_id, -1)]
        elif isinstance(user_id, list):
            user_idx = [self.recom.uid_map.get(uid, -1) for uid in user_id]

        if any(i == -1 for i in user_idx):
            raise ValueError(f"{user_id} is unknown to the model.")

        if k < -1 or k > self.recom.total_items:
            raise ValueError(
                f"k={k} is invalid, there are {self.recom.total_users} users in total."
            )

        query = self.recom.get_user_query(user_idx)
        knn_items = self.knn_query(query, k=k)

        # TODO: remove seen items

        recommendations = [
            [self.recom.item_ids[i] for i in knn_items[u]] for u in range(len(user_idx))
        ]
        return recommendations

    def save(self, save_dir=None):
        # TODO: implement save recom and index
        raise NotImplementedError()

    @staticmethod
    def load(model_path, trainable=False):
        # TODO: implement load recom and index
        raise NotImplementedError()
