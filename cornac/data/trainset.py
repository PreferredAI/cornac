# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from scipy.sparse import csr_matrix


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

    def is_known_user(self, user_id):
        return user_id in self._uid_map

    def is_known_item(self, item_id):
        return item_id in self._iid_map

    def map_uid(self, user_id):
        mapped_uid = self._uid_map[user_id]
        return mapped_uid

    def map_iid(self, item_id):
        mapped_iid = self._iid_map[item_id]
        return mapped_iid


class MatrixTrainSet(TrainSet):

    def __init__(self, matrix, uid_map, iid_map):
        TrainSet.__init__(self, uid_map, iid_map)
        self._matrix = matrix

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix = matrix

    @classmethod
    def from_triplets(cls, triplet_data):
        uid_map = {}
        iid_map = {}

        u_indices = []
        i_indices = []
        r_values = []
        for user, item, rating in triplet_data:
            mapped_uid = uid_map.setdefault(user, len(uid_map))
            mapped_iid = iid_map.setdefault(item, len(iid_map))

            u_indices.append(mapped_uid)
            i_indices.append(mapped_iid)
            r_values.append(rating)

        # csr_matrix is more efficient for row (user) slicing
        csr_mat = csr_matrix((r_values, (u_indices, i_indices)), shape=(len(uid_map), len(iid_map)))

        return cls(csr_mat, uid_map, iid_map)
