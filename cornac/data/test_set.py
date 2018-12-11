# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""


class TestSet:

    def __init__(self, user_ratings):
        self._user_ratings = user_ratings

    def get_users(self):
        return self._user_ratings.keys()

    def get_ratings(self, user_id):
        return self._user_ratings.get(user_id, [])

    @classmethod
    def from_triplets(self, triplet_data):
        user_ratings = {}

        for user, item, rating in triplet_data:
            if user not in user_ratings:
                user_ratings[user] = []
            user_ratings[user].append((item, rating))

        return self(user_ratings)
