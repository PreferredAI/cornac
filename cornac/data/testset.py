# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from collections import OrderedDict


class TestSet:

    def __init__(self, user_ratings, uid_map, iid_map):
        self._user_ratings = user_ratings
        self._uid_map = uid_map
        self._iid_map = iid_map


    def get_users(self):
        return self._user_ratings.keys()


    def get_ratings(self, mapped_uid):
        return self._user_ratings.get(mapped_uid, [])


    def get_uid(self, raw_uid):
        return self._uid_map[raw_uid]


    def get_iid(self, raw_iid):
        return self._iid_map[raw_iid]



    @classmethod
    def from_triplets(self, triplet_data, pre_uid_map, pre_iid_map, pre_ur_set, verbose=False):
        uid_map = OrderedDict()
        iid_map = OrderedDict()
        user_ratings = {}

        for raw_uid, raw_iid, rating in triplet_data:
            if (raw_uid, raw_iid) in pre_ur_set: # duplicate rating
                continue
            pre_ur_set.add((raw_uid, raw_iid))

            mapped_uid = pre_uid_map.setdefault(raw_uid, len(pre_uid_map))
            mapped_iid = pre_iid_map.setdefault(raw_iid, len(pre_iid_map))
            uid_map[raw_uid] = mapped_uid
            iid_map[raw_iid] = mapped_iid

            ur_list = user_ratings.setdefault(mapped_uid, [])
            ur_list.append((mapped_iid, float(rating)))

        if verbose:
            print('Number of tested users = {}'.format(len(user_ratings)))

        return self(user_ratings, uid_map, iid_map)
