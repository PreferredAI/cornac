# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from scipy.sparse import csr_matrix
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


class MatrixTrainSet(TrainSet):

    def __init__(self, matrix, max_rating, min_rating, global_mean, uid_map, iid_map):
        TrainSet.__init__(self, uid_map, iid_map)
        self.matrix = matrix
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.global_mean = global_mean
        self.item_ppl_rank = self._rank_items_by_popularity(matrix)


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
    def from_uir_triplets(cls, triplet_data, pre_uid_map, pre_iid_map, pre_ui_set, verbose=False):
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
            if (raw_uid, raw_iid) in pre_ui_set: # duplicate rating
                continue
            pre_ui_set.add((raw_uid, raw_iid))

            mapped_uid = pre_uid_map.setdefault(raw_uid, len(pre_uid_map))
            mapped_iid = pre_iid_map.setdefault(raw_iid, len(pre_iid_map))
            uid_map[raw_uid] = mapped_uid
            iid_map[raw_iid] = mapped_iid

            rating = float(rating)
            rating_sum += rating
            rating_count +=1
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
