import torch
import torch.utils.data as data
from cornac.data.dataset import Dataset
import numpy as np


class PWLearningSampler(data.Dataset):
    def __init__(self, cornac_dataset: Dataset, num_neg: int):
        self.data = cornac_dataset
        self.num_neg = num_neg
        # make sure we only have positive ratings no unseen interactions
        assert np.all(self.data.uir_tuple[2] > 0)
        self.user_array = self.data.uir_tuple[0]
        self.item_array = self.data.uir_tuple[1]

    def __getitems__(self, list_of_indexs):
        """
        vectorized version of __getitem__
        """
        batch_size = len(list_of_indexs)
        users = self.user_array[list_of_indexs]
        pos_items = self.item_array[list_of_indexs]

        pos_u_i = np.vstack([users, pos_items]).T

        neg_examples_idxs = np.random.choice(len(self.data.uir_tuple[1]), (batch_size, self.num_neg))
        while True:

            filter = np.any(users.reshape(len(users), 1) == self.user_array[neg_examples_idxs], axis=1)
            if np.any(filter):
                num_pos_examples = neg_examples_idxs[~filter].shape[0]
                new_neg_examples_idx = np.random.choice(len(self.data.uir_tuple[1]), (num_pos_examples, self.num_neg))
                neg_examples_idxs[~filter] = new_neg_examples_idx
            else:
                # draw the neg examples
                neg_i = self.item_array[neg_examples_idxs]
                break

        return np.hstack([pos_u_i, neg_i])


    def __getitem__(self, index):

        # first select index tuple
        user = self.user_array[index]
        item = self.item_array[index]

        # now construct positive case
        pos_u_i = [user, item]

        # samle num_neg manu negative case, have to make sure its not a pos case
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
        return len(self.data.uir_tuple[0])

# for batch in dataloader:
#     print(batch)