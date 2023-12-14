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

import warnings
from time import time

import numpy as np
from tqdm import tqdm

from ..recommender import NextBasketRecommender


class TIFUKNN(NextBasketRecommender):
    """Temporal-Item-Frequency-based User-KNN (TIFUKNN)

    Parameters
    ----------
    name: string, default: 'TIFUKNN'
        The name of the recommender model.

    n_neighbors: int, optional, default: 300
        The number of neighbors for KNN

    within_decay_rate: float, optional, default: 0.9
        Within-basket time-decayed ratio in range [0, 1]

    group_decay_rate: float, optional, default: 0.7
        Group time-decayed ratio in range [0, 1]

    alpha: float, optional, default: 0.7
        The trade-off between current user vector and neighbors vectors
        to compute final item scores

    n_groups: int, optional, default: 7
        The historal baskets will be partition into `n_groups` equally.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    References
    ----------
    Haoji Hu, Xiangnan He, Jinyang Gao, and Zhi-Li Zhang. 2020.
    Modeling Personalized Item Frequency Information for Next-basket Recommendation.
    In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '20). Association for Computing Machinery, New York, NY, USA, 1071–1080. https://doi.org/10.1145/3397271.3401066

    """

    def __init__(
        self,
        name="TIFUKNN",
        n_neighbors=300,
        within_decay_rate=0.9,
        group_decay_rate=0.7,
        alpha=0.7,
        n_groups=7,
        verbose=False,
    ):
        super().__init__(name=name, trainable=False, verbose=verbose)
        assert within_decay_rate >= 0 and within_decay_rate <= 1
        assert group_decay_rate >= 0 and group_decay_rate <= 1
        self.n_neighbors = n_neighbors
        self.within_decay_rate = within_decay_rate
        self.group_decay_rate = group_decay_rate
        self.alpha = alpha
        self.n_groups = n_groups

    def fit(self, train_set, val_set=None):
        from scipy.spatial import KDTree

        super().fit(train_set=train_set, val_set=val_set)
        self.user_vectors = self._get_user_vectors(self.train_set)
        if self.n_neighbors > len(self.user_vectors):
            warnings.warn("Number of users is %d, smaller than number of neighbors %d" % (len(self.user_vectors), self.n_neighbors))
            self.n_neighbors = len(self.user_vectors)

        start_time = time()
        if self.verbose:
            print("Constructing kd-tree for quick nearest-neighbor lookup")
        self.tree = KDTree(self.user_vectors)
        if self.verbose:
            print("Constructing kd-tree for quick nearest-neighbor lookup takes %.0f" % (time() - start_time))
        return self

    def _get_user_vectors(self, data_set):
        user_vectors = []
        for _, _, [basket_items] in tqdm(
            data_set.ubi_iter(batch_size=1, shuffle=False),
            desc="Getting user vectors",
            total=data_set.num_users,
        ):
            user_vectors.append(self._compute_user_vector(basket_items[:-1]))
        user_vectors = np.asarray(user_vectors, dtype="float32")
        return user_vectors

    def _compute_user_vector(self, history_baskets):
        his_list = []
        n_baskets = len(history_baskets)
        for inc, iids in enumerate(history_baskets):
            his_vec = np.zeros(self.total_items, dtype="float32")
            decayed_val = np.power(self.within_decay_rate, n_baskets - inc - 1)
            for iid in iids:
                his_vec[iid] = decayed_val
            his_list.append(his_vec)
        grouped_list, real_n_groups = self._group_history_list(his_list, self.n_groups)
        his_vec = np.zeros(self.total_items, dtype="float32")
        if real_n_groups == 0:
            return his_vec

        for idx in range(real_n_groups):
            decayed_val = np.power(self.group_decay_rate, self.n_groups - idx - 1)
            his_vec += grouped_list[idx] * decayed_val

        return his_vec / real_n_groups

    def _group_history_list(self, his_list, n_groups):
        grouped_vec_list = []
        if len(his_list) < n_groups:
            for j in range(len(his_list)):
                grouped_vec_list.append(his_list[j])
            return grouped_vec_list, len(his_list)
        else:
            est_num_vec_each_block = len(his_list) / n_groups
            base_num_vec_each_block = int(np.floor(len(his_list) / n_groups))
            residual = est_num_vec_each_block - base_num_vec_each_block

            num_vec_has_extra_vec = int(np.round(residual * n_groups))

            if residual == 0:
                for i in range(n_groups):
                    sum = np.zeros(len(his_list[0]))
                    for j in range(base_num_vec_each_block):
                        sum += his_list[i * base_num_vec_each_block + j]
                    grouped_vec_list.append(sum / base_num_vec_each_block)
            else:
                for i in range(n_groups - num_vec_has_extra_vec):
                    sum = np.zeros(len(his_list[0]))
                    for j in range(base_num_vec_each_block):
                        sum += his_list[i * base_num_vec_each_block + j]
                        last_idx = i * base_num_vec_each_block + j
                    grouped_vec_list.append(sum / base_num_vec_each_block)

                est_num = int(np.ceil(est_num_vec_each_block))
                start_group_idx = n_groups - num_vec_has_extra_vec
                if len(his_list) - start_group_idx * base_num_vec_each_block >= est_num_vec_each_block:
                    for i in range(start_group_idx, n_groups):
                        sum = np.zeros(len(his_list[0]))
                        for j in range(est_num):
                            iidxx = last_idx + 1 + (i - start_group_idx) * est_num + j
                            sum += his_list[iidxx]
                        grouped_vec_list.append(sum / est_num)

            return grouped_vec_list, n_groups

    def score(self, user_idx, history_baskets, **kwargs):
        if len(history_baskets) == 0:
            return np.zeros(self.total_items, dtype="float32")
        user_vector = self._compute_user_vector(history_baskets)
        _, indices = self.tree.query([user_vector], k=self.n_neighbors)
        return self.alpha * user_vector + (1 - self.alpha) * np.mean(self.user_vectors[indices.squeeze()])
