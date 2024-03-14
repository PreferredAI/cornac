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

from typing import List
import torch.utils.data as data
from cornac.data.dataset import Dataset
import numpy as np


class PWLearningSampler(data.Dataset):
    """
    Sampler for pairwised based ranking loss function. This sampler will return
    a batch of positive and negative items. Per user index there will be one
    positive and num_neg negative items.
    This sampler is to be used in the PyTorch DatLoader, through which loading
    can be distributed amongst multiple workers.
    """
    def __init__(self, cornac_dataset: Dataset, num_neg: int):
        self.data = cornac_dataset
        self.num_neg = num_neg
        # make sure we only have positive ratings no unseen interactions
        assert np.all(self.data.uir_tuple[2] > 0)
        self.user_array = self.data.uir_tuple[0]

        self.item_array = self.data.uir_tuple[1]
        self.unique_items = np.unique(self.item_array)
        self.unique_users = np.unique(self.user_array)
        self.user_item_array = np.vstack([self.user_array, self.item_array]).T
        # make sure users are assending from 0
        np.all(np.unique(self.user_array) == np.arange(self.unique_users.shape[0]))

    def __getitems__(self, list_of_indexs: List[int]):
        """
        Vectorized version of __getitem__

        Uses list_of_indixes to index into uir_tuple from cornac dataset and
        thus retrieve 1 positive item per given user. Additionally random
        samples num_neg negative items per user given in list_of_indices.

        Parameters
        ----------

        list_of_indexs: List[int]
            List of indexs to sample from the uir_tuple given in cornac dataset
        """
        batch_size = len(list_of_indexs)
        users = self.user_array[list_of_indexs]
        pos_items = self.item_array[list_of_indexs]

        pos_u_i = np.vstack([users, pos_items]).T

        # sample negative items per user
        neg_item_list = []
        for _ in range(self.num_neg):
            neg_items = np.random.choice(self.data.csr_matrix.shape[1], batch_size)
            # make sure we dont sample a positive item
            candidates = self.data.csr_matrix[users, neg_items]
            while candidates.nonzero()[0].size != 0:
                replacement_neg_items = np.random.choice(self.data.csr_matrix.shape[1], candidates.nonzero()[0].size)
                neg_items[candidates.nonzero()[1]] = replacement_neg_items
                candidates = self.data.csr_matrix[users, neg_items]
            
            neg_item_list.append(neg_items)
        neg_items = np.vstack(neg_item_list).T
        return np.hstack([pos_u_i, neg_items])

    def __getitem__(self, index):
        """
        Uses index into uir_tuple from cornac dataset and
        thus retrieves 1 positive user-item pair. Additionally random
        samples num_neg negative items for that user.

        Parameters
        ----------

        list_of_indexs: List[int]
            List of indexs to sample from the uir_tuple given in cornac dataset
        """
        # first select index tuple
        user = self.user_array[index]
        item = self.item_array[index]

        # now construct positive case
        pos_u_i = [user, item]

        i = 0
        neg_i = []
        while i < self.num_neg:
            neg_example = np.random.choice(self.data.uir_tuple[1])

            idxs_of_item = np.where(self.item_array == neg_example)
            users_who_have_rated_item = self.user_array[idxs_of_item]

            if user not in users_who_have_rated_item:
                i += 1
                neg_i = neg_i + [neg_example]
        # return user, item_positive, num_neg * item_neg array
        return np.array(pos_u_i + neg_i)

    def __len__(self):
        """
        Return length of sampler as length of uir_tuple from cornac dataset.
        """
        return len(self.data.uir_tuple[0])
