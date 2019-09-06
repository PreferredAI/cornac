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

from collections import OrderedDict, defaultdict

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix

from ..utils import get_rng
from ..utils import estimate_batches


class Dataset(object):
    """Training set contains preference matrix

    Parameters
    ----------
    uid_map: :obj:`OrderDict`
        The dictionary containing mapping from original ids to mapped ids of users.

    iid_map: :obj:`OrderDict`
        The dictionary containing mapping from original ids to mapped ids of items.

    uir_tuple: tuple
        Tuple of 3 numpy arrays (users, items, ratings).

    max_rating: float
        Maximum value of the preferences.

    min_rating: float
        Minimum value of the preferences.

    global_mean: float
        Average value of the preferences.

    seed: int, optional, default: None
        Random seed for reproducing data sampling.

    """

    def __init__(self, uid_map, iid_map, uir_tuple, num_users, num_items,
                 max_rating, min_rating, global_mean, seed=None):
        self.uid_map = uid_map
        self.iid_map = iid_map
        self.uir_tuple = uir_tuple
        self.num_users = num_users
        self.num_items = num_items
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.global_mean = global_mean
        self.rng = get_rng(seed)

        self.__user_ids = None
        self.__item_ids = None
        self.__user_indices = None
        self.__item_indices = None

        self.__user_data = None
        self.__item_data = None
        self.__csr_matrix = None
        self.__csc_matrix = None
        self.__dok_matrix = None

    @property
    def user_ids(self):
        """Return a list of the raw user ids"""
        if self.__user_ids is None:
            self.__user_ids = list(self.uid_map.keys())
        return self.__user_ids

    @property
    def item_ids(self):
        """Return a list of the raw item ids"""
        if self.__item_ids is None:
            self.__item_ids = list(self.iid_map.keys())
        return self.__item_ids

    @property
    def user_indices(self):
        """Return a numpy array of the user indices"""
        if self.__user_indices is None:
            self.__user_indices = np.fromiter(self.uid_map.values(), dtype=np.int)
        return self.__user_indices

    @property
    def item_indices(self):
        """Return a numpy array of the item indices"""
        if self.__item_indices is None:
            self.__item_indices = np.fromiter(self.iid_map.values(), dtype=np.int)
        return self.__item_indices

    @property
    def uir_tuple(self):
        """Return tuple of three numpy arrays (users, items, ratings)"""
        return self.__uir_tuple

    @uir_tuple.setter
    def uir_tuple(self, input_tuple):
        if input_tuple is not None and len(input_tuple) != 3:
            raise ValueError('input_tuple required to be size 3 but size {}'.format(len(input_tuple)))
        self.__uir_tuple = input_tuple

    @property
    def user_data(self):
        """Return user-oriented data. Each user contains a tuple of two lists (items, ratings)"""
        if self.__user_data is None:
            self.__user_data = defaultdict()
            for u, i, r in zip(*self.uir_tuple):
                u_data = self.__user_data.setdefault(u, ([], []))
                u_data[0].append(i)
                u_data[1].append(r)
        return self.__user_data

    @property
    def item_data(self):
        """Return item-oriented data. Each item contains a tuple of two lists (users, ratings)"""
        if self.__item_data is None:
            self.__item_data = defaultdict()
            for u, i, r in zip(*self.uir_tuple):
                i_data = self.__item_data.setdefault(i, ([], []))
                i_data[0].append(u)
                i_data[1].append(r)
        return self.__item_data

    @property
    def matrix(self):
        """Return the user-item interaction matrix in csr sparse format"""
        return self.csr_matrix

    @property
    def csr_matrix(self):
        """Return the user-item interaction matrix in CSR sparse format"""
        if self.__csr_matrix is None:
            (u_indices, i_indices, r_values) = self.uir_tuple
            self.__csr_matrix = csr_matrix((r_values, (u_indices, i_indices)),
                                           shape=(self.num_users, self.num_items))
        return self.__csr_matrix

    @property
    def csc_matrix(self):
        """Return the user-item interaction matrix in CSC sparse format"""
        if self.__csc_matrix is None:
            (u_indices, i_indices, r_values) = self.uir_tuple
            self.__csc_matrix = csc_matrix((r_values, (u_indices, i_indices)),
                                           shape=(self.num_users, self.num_items))
        return self.__csc_matrix

    @property
    def dok_matrix(self):
        """Return the user-item interaction matrix in DOK sparse format"""
        if self.__dok_matrix is None:
            self.__dok_matrix = dok_matrix((self.num_users, self.num_items), dtype=np.float32)
            for u, i, r in zip(*self.uir_tuple):
                self.__dok_matrix[u, i] = r
        return self.__dok_matrix

    @classmethod
    def from_uir(cls, data, global_uid_map=None, global_iid_map=None,
                 global_ui_set=None, seed=None, exclude_unknowns=False):
        """Constructing Dataset from UIR triplet data.

        Parameters
        ----------
        data: array-like, shape: [n_examples, 3]
            Data in the form of triplets (user, item, rating)

        global_uid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of users.

        global_iid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of items.

        global_ui_set: :obj:`set`, optional, default: None
            The global set of tuples (user, item). This helps avoiding duplicate observations.

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        exclude_unknowns: bool, default: False
            Ignore unknown users and items (cold-start).

        Returns
        -------
        res: :obj:`<cornac.data.Dataset>`
            Dataset object.

        """
        if global_uid_map is None:
            global_uid_map = OrderedDict()
        if global_iid_map is None:
            global_iid_map = OrderedDict()
        if global_ui_set is None:
            global_ui_set = set()

        uid_map = OrderedDict()
        iid_map = OrderedDict()

        u_indices = []
        i_indices = []
        r_values = []

        rating_sum = 0.
        rating_count = 0
        max_rating = float('-inf')
        min_rating = float('inf')

        filtered_data = [(u, i, r) for u, i, r in data if (u, i) not in global_ui_set]
        if exclude_unknowns:
            filtered_data = [(u, i, r) for u, i, r in filtered_data
                             if u in global_uid_map and i in global_iid_map]

        for uid, iid, rating in filtered_data:
            global_ui_set.add((uid, iid))
            uid_map[uid] = global_uid_map.setdefault(uid, len(global_uid_map))
            iid_map[iid] = global_iid_map.setdefault(iid, len(global_iid_map))

            rating = float(rating)
            rating_sum += rating
            rating_count += 1
            if rating > max_rating:
                max_rating = rating
            if rating < min_rating:
                min_rating = rating

            u_indices.append(uid_map[uid])
            i_indices.append(iid_map[iid])
            r_values.append(rating)

        num_users = len(global_uid_map)
        num_items = len(global_iid_map)
        global_mean = rating_sum / rating_count

        uir_tuple = (np.asarray(u_indices, dtype=np.int),
                     np.asarray(i_indices, dtype=np.int),
                     np.asarray(r_values, dtype=np.float))

        return cls(uid_map=uid_map, iid_map=iid_map, uir_tuple=uir_tuple, num_users=num_users, num_items=num_items,
                   max_rating=max_rating, min_rating=min_rating, global_mean=global_mean, seed=seed)

    def num_batches(self, batch_size):
        return estimate_batches(len(self.uir_tuple[0]), batch_size)

    def idx_iter(self, idx_range, batch_size=1, shuffle=False):
        """Create an iterator over batch of indices

        Parameters
        ----------
        batch_size : int, optional, default = 1

        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of indices (array of np.int)

        """
        indices = np.arange(idx_range)
        if shuffle:
            self.rng.shuffle(indices)

        n_batches = estimate_batches(len(indices), batch_size)
        for b in range(n_batches):
            start_offset = batch_size * b
            end_offset = batch_size * b + batch_size
            end_offset = min(end_offset, len(indices))
            batch_ids = indices[start_offset:end_offset]
            yield batch_ids

    def uir_iter(self, batch_size=1, shuffle=False, num_zeros=0):
        """Create an iterator over data yielding batch of users, items, and rating values

        Parameters
        ----------
        batch_size : int, optional, default = 1

        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        num_zeros : int, optional, default = 0
            Number of unobserved ratings (zeros) to be added per user. This could be used
            for negative sampling. By default, no values are added.

        Returns
        -------
        iterator : batch of users (array of np.int), batch of items (array of np.int),
            batch of ratings (array of np.float)

        """
        for batch_ids in self.idx_iter(len(self.uir_tuple[0]), batch_size, shuffle):
            batch_users = self.uir_tuple[0][batch_ids]
            batch_items = self.uir_tuple[1][batch_ids]
            batch_ratings = self.uir_tuple[2][batch_ids]

            if num_zeros > 0:
                repeated_users = batch_users.repeat(num_zeros)
                neg_items = np.empty_like(repeated_users)
                for i, u in enumerate(repeated_users):
                    j = self.rng.randint(0, self.num_items)
                    while self.dok_matrix[u, j] > 0:
                        j = self.rng.randint(0, self.num_items)
                    neg_items[i] = j
                batch_users = np.concatenate((batch_users, repeated_users))
                batch_items = np.concatenate((batch_items, neg_items))
                batch_ratings = np.concatenate((batch_ratings, np.zeros_like(neg_items)))

            yield batch_users, batch_items, batch_ratings

    def uij_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over data yielding batch of users, positive items, and negative items

        Parameters
        ----------
        batch_size : int, optional, default = 1

        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of users (array of np.int), batch of positive items (array of np.int),
            batch of negative items (array of np.int)

        """
        for batch_ids in self.idx_iter(len(self.uir_tuple[0]), batch_size, shuffle):
            batch_users = self.uir_tuple[0][batch_ids]
            batch_pos_items = self.uir_tuple[1][batch_ids]
            batch_pos_ratings = self.uir_tuple[2][batch_ids]
            batch_neg_items = np.zeros_like(batch_pos_items)
            for i, (user, pos_rating) in enumerate(zip(batch_users, batch_pos_ratings)):
                neg_item = self.rng.randint(0, self.num_items)
                while self.dok_matrix[user, neg_item] >= pos_rating:
                    neg_item = self.rng.randint(0, self.num_items)
                batch_neg_items[i] = neg_item
            yield batch_users, batch_pos_items, batch_neg_items

    def user_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over user indices

        Parameters
        ----------
        batch_size : int, optional, default = 1

        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of user indices (array of np.int)
        """
        for batch_ids in self.idx_iter(len(self.user_indices), batch_size, shuffle):
            yield self.user_indices[batch_ids]

    def item_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over item indices

        Parameters
        ----------
        batch_size : int, optional, default = 1

        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of item indices (array of np.int)
        """
        for batch_ids in self.idx_iter(len(self.item_indices), batch_size, shuffle):
            yield self.item_indices[batch_ids]

    def is_unk_user(self, user_idx):
        """Return whether or not a user is unknown given the user index"""
        return user_idx >= self.num_users

    def is_unk_item(self, item_idx):
        """Return whether or not an item is unknown given the item index"""
        return item_idx >= self.num_items

    def add_modalities(self, **kwargs):
        self.user_text = kwargs.get('user_text', None)
        self.item_text = kwargs.get('item_text', None)
        self.user_image = kwargs.get('user_image', None)
        self.item_image = kwargs.get('item_image', None)
        self.user_graph = kwargs.get('user_graph', None)
        self.item_graph = kwargs.get('item_graph', None)