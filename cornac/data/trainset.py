# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from scipy.sparse import csr_matrix, find
from collections import OrderedDict
import numpy as np


class TrainSet:

    def __init__(self, uid_map, iid_map):
        self._uid_map = uid_map
        self._iid_map = iid_map

    @property
    def num_users(self):
        return len(self._uid_map)

    @property
    def num_items(self):
        return len(self._iid_map)

    def is_unk_user(self, mapped_uid):
        return mapped_uid >= self.num_users

    def is_unk_item(self, mapped_iid):
        return mapped_iid >= self.num_items

    def get_uid(self, raw_uid):
        return self._uid_map[raw_uid]

    def get_iid(self, raw_iid):
        return self._iid_map[raw_iid]

    def get_uid_list(self):
        return self._uid_map.values()

    def get_raw_uid_list(self):
        return self._uid_map.keys()

    def get_iid_list(self):
        return self._iid_map.values()

    def get_raw_iid_list(self):
        return self._iid_map.keys()

    def idx_iter(self, idx_range, batch_size=1, shuffle=False):
        """ Create an iterator over batch of indices

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
            np.random.shuffle(indices)

        n_batches = int(np.ceil(len(indices) / batch_size))
        for b in range(n_batches):
            start_offset = batch_size * b
            end_offset = batch_size * b + batch_size
            end_offset = min(end_offset, len(indices))

            batch_ids = indices[start_offset:end_offset]
            yield batch_ids


class MatrixTrainSet(TrainSet):

    def __init__(self, matrix, max_rating, min_rating, global_mean, uid_map, iid_map):
        TrainSet.__init__(self, uid_map, iid_map)
        self.matrix = matrix
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.global_mean = global_mean
        self.item_ppl_rank = self._rank_items_by_popularity(matrix)
        self.triplets = None

    @property
    def num_users(self):
        return self.matrix.shape[0]

    @property
    def num_items(self):
        return self.matrix.shape[1]

    @staticmethod
    def _rank_items_by_popularity(rating_matrix):
        item_ppl_scores = rating_matrix.sum(axis=0)
        item_rank = np.argsort(item_ppl_scores.A1)[::-1]
        return item_rank

    @classmethod
    def from_uir_triplets(cls, triplet_data, pre_uid_map=None, pre_iid_map=None,
                          pre_ui_set=None, verbose=False):
        if pre_uid_map is None:
            pre_uid_map = OrderedDict()
        if pre_iid_map is None:
            pre_iid_map = OrderedDict()
        if pre_ui_set is None:
            pre_ui_set = set()

        uid_map = OrderedDict()
        iid_map = OrderedDict()

        u_indices = []
        i_indices = []
        r_values = []

        rating_sum = 0.
        rating_count = 0
        max_rating = float('-inf')
        min_rating = float('inf')

        for raw_uid, raw_iid, rating in triplet_data:
            if (raw_uid, raw_iid) in pre_ui_set:  # duplicate rating
                continue
            pre_ui_set.add((raw_uid, raw_iid))

            mapped_uid = pre_uid_map.setdefault(raw_uid, len(pre_uid_map))
            mapped_iid = pre_iid_map.setdefault(raw_iid, len(pre_iid_map))
            uid_map[raw_uid] = mapped_uid
            iid_map[raw_iid] = mapped_iid

            rating = float(rating)
            rating_sum += rating
            rating_count += 1
            if rating > max_rating:
                max_rating = rating
            if rating < min_rating:
                min_rating = rating

            u_indices.append(mapped_uid)
            i_indices.append(mapped_iid)
            r_values.append(rating)

        # csr_matrix is more efficient for row (user) slicing
        csr_mat = csr_matrix((r_values, (u_indices, i_indices)), shape=(len(uid_map), len(iid_map)))
        global_mean = rating_sum / rating_count

        if verbose:
            print('Number of training users = {}'.format(len(uid_map)))
            print('Number of training items = {}'.format(len(iid_map)))
            print('Max rating = {:.1f}'.format(max_rating))
            print('Min rating = {:.1f}'.format(min_rating))
            print('Global mean = {:.1f}'.format(global_mean))

        return cls(csr_mat, max_rating, min_rating, global_mean, uid_map, iid_map)

    def uir_iter(self, batch_size=1, shuffle=False):
        """ Create an iterator over data yielding batch of users, items, and rating values

        Parameters
        ----------
        batch_size : int, optional, default = 1
        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of users (array of np.int), batch of items (array of np.int),
            batch of ratings (array of np.float)
        """
        if self.triplets is None:
            self.triplets = find(self.matrix)

        for batch_ids in self.idx_iter(len(self.triplets[0]), batch_size, shuffle):
            batch_users = self.triplets[0][batch_ids]
            batch_items = self.triplets[1][batch_ids]
            batch_ratings = self.triplets[2][batch_ids]

            yield batch_users, batch_items, batch_ratings

    def uij_iter(self, batch_size=1, shuffle=False):
        """ Create an iterator over data yielding batch of users, positive items, and negative items

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
        if self.triplets is None:
            self.triplets = find(self.matrix)

        for batch_ids in self.idx_iter(len(self.triplets[0]), batch_size, shuffle):
            batch_users = self.triplets[0][batch_ids]
            batch_pos_items = self.triplets[1][batch_ids]
            batch_pos_ratings = self.triplets[2][batch_ids]

            batch_neg_items = np.zeros_like(batch_pos_items)
            for i, user, pos_rating in enumerate(zip(batch_users, batch_pos_ratings)):
                neg_item = np.random.randint(0, self.num_items - 1)
                while self.matrix[user, neg_item] >= pos_rating:
                    neg_item = np.random.randint(0, self.num_items - 1)
                batch_neg_items[i] = neg_item

            yield batch_users, batch_pos_items, batch_neg_items
