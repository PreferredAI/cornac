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


    def is_known_user(self, mapped_uid):
        return mapped_uid < self.num_users


    def is_known_item(self, mapped_iid):
        return mapped_iid < self.num_items


    def get_uid(self, raw_uid):
        return self._uid_map[raw_uid]


    def get_iid(self, raw_iid):
        return self._iid_map[raw_iid]


    def get_uid_list(self):
        return self._uid_map.values()


    def get_iid_list(self):
        return self._iid_map.values()


class MatrixTrainSet(TrainSet):

    def __init__(self, matrix, max_rating, min_rating, global_mean, uid_map, iid_map):
        TrainSet.__init__(self, uid_map, iid_map)

        self.matrix = matrix
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.global_mean = global_mean

        print('Number of training users = {}'.format(self.num_users))
        print('Number of training users = {}'.format(self.num_items))
        print('Max rating = {:.1f}'.format(self.max_rating))
        print('Min rating = {:.1f}'.format(self.min_rating))
        print('Global mean = {:.1f}'.format(self.global_mean))


    @classmethod
    def from_triplets(cls, triplet_data, pre_uid_map, pre_iid_map, pre_ur_set):
        uid_map = {}
        iid_map = {}

        u_indices = []
        i_indices = []
        r_values = []
        rating_sum = 0.
        rating_count = 0

        for raw_uid, raw_iid, rating in triplet_data:
            if (raw_uid, raw_iid) in pre_ur_set: # duplicate rating
                continue
            pre_ur_set.add((raw_uid, raw_iid))

            mapped_uid = pre_uid_map.setdefault(raw_uid, len(pre_uid_map))
            mapped_iid = pre_iid_map.setdefault(raw_iid, len(pre_iid_map))
            uid_map[raw_uid] = mapped_uid
            iid_map[raw_iid] = mapped_iid

            rating = float(rating)
            rating_sum += rating
            rating_count +=1

            u_indices.append(mapped_uid)
            i_indices.append(mapped_iid)
            r_values.append(rating)

        # csr_matrix is more efficient for row (user) slicing
        csr_mat = csr_matrix((r_values, (u_indices, i_indices)), shape=(len(uid_map), len(iid_map)))
        global_mean = rating_sum / rating_count

        return cls(csr_mat, global_mean, uid_map, iid_map)
