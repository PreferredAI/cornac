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
        self.unique_items = np.unique(self.item_array)
        self.unique_users = np.unique(self.user_array)
        self.user_item_array = np.vstack([self.user_array, self.item_array]).T
        # make sure users are assending from 0
        np.all(np.unique(self.user_array) == np.arange(self.unique_users.shape[0]))

        self.precompute_neg_items_per_user()


    def precompute_neg_items_per_user(self):
        """
        Precompute negative items for each user so we can dynamically sample from this list
        """
        # precompute per user list of negative items
        self.user_neg_items = []
        for user in self.unique_users:
            idxs_of_user = np.where(self.user_array == user)
            items_of_user = self.item_array[idxs_of_user]
            # get difference between all items and items of user
            neg_items = np.setdiff1d(self.unique_items, items_of_user)
            self.user_neg_items.append(neg_items)

        # now fill -1 to make all user_neg_items the same length
        max_len = max([len(x) for x in self.user_neg_items])
        for i in range(len(self.user_neg_items)):
            self.user_neg_items[i] = np.pad(self.user_neg_items[i], (0, max_len - len(self.user_neg_items[i])), 'constant', constant_values=-1)
        self.user_neg_items = np.array(self.user_neg_items)

    def __getitems__(self, list_of_indexs):
        """
        vectorized version of __getitem__
        """
        batch_size = len(list_of_indexs)
        users = self.user_array[list_of_indexs]
        pos_items = self.item_array[list_of_indexs]

        pos_u_i = np.vstack([users, pos_items]).T

        users_neg_items = self.user_neg_items[pos_u_i[:,0]]
        # sample negative items
        random_idxs = np.random.choice(range(0, users_neg_items.shape[1] - 1), batch_size * self.num_neg * 2)
        selected_neg_items_per_user = users_neg_items[list(range(users_neg_items.shape[0])) * self.num_neg*2, random_idxs]
        selected_neg_items_per_user = selected_neg_items_per_user.reshape(batch_size, self.num_neg*2)
        
        for i in range(2):
            filter = (selected_neg_items_per_user[:, :self.num_neg] == -1)
            if np.any(filter):
                problem_candidates = np.where(selected_neg_items_per_user[:, :self.num_neg] == -1)
                # draw the neg examples from the rest of the previously sampled neg examples
                neg_items = selected_neg_items_per_user[problem_candidates[0], self.num_neg:]
                selected_neg_items_per_user[problem_candidates[0], problem_candidates[1]] = neg_items[:, i]
            else:
                # draw the neg examples
                break
        assert np.all(selected_neg_items_per_user[:, :self.num_neg] != -1)
        return np.hstack([pos_u_i, selected_neg_items_per_user[:, :self.num_neg]])


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