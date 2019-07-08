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

from collections import OrderedDict


class TestSet:
    """Test Set

    Parameters
    ----------
    user_ratings: :obj:`defaultdict` of :obj:`list`
        The dictionary containing lists of tuples of the form (item, rating). The keys are user ids.

    uid_map: :obj:`defaultdict`
        The dictionary containing mapping from original ids to mapped ids of users.

    iid_map: :obj:`defaultdict`
        The dictionary containing mapping from original ids to mapped ids of items.

    """

    def __init__(self, user_ratings, uid_map, iid_map):
        self._user_ratings = user_ratings
        self._uid_map = uid_map
        self._iid_map = iid_map

    @property
    def users(self):
        """Return a list of users"""
        return list(self._user_ratings.keys())

    def get_ratings(self, mapped_uid):
        """Return a list of tuples of (item, rating) of given mapped user id"""
        return self._user_ratings.get(mapped_uid, [])

    def get_uid(self, raw_uid):
        """Return the mapped id of a user given a raw id"""
        return self._uid_map[raw_uid]

    def get_iid(self, raw_iid):
        """Return the mapped id of an item given a raw id"""
        return self._iid_map[raw_iid]

    @classmethod
    def from_uir(self, data, global_uid_map, global_iid_map, global_ui_set, verbose=False):
        """Constructing TestSet from triplet data.

        Parameters
        ----------
        data: array-like, shape: [n_examples, 3]
            Data in the form of triplets (user, item, rating)

        global_uid_map: :obj:`defaultdict`
            The dictionary containing global mapping from original ids to mapped ids of users.

        global_iid_map: :obj:`defaultdict`
            The dictionary containing global mapping from original ids to mapped ids of items.

        global_ui_set: :obj:`set`
            The global set of tuples (user, item). This helps avoiding duplicate observations.

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        test_set: :obj:`<cornac.data.TestSet>`
            TestSet object.

        """

        uid_map = OrderedDict()
        iid_map = OrderedDict()
        user_ratings = {}

        unk_user_count = 0
        unk_item_count = 0

        for raw_uid, raw_iid, rating in data:
            if (raw_uid, raw_iid) in global_ui_set:  # duplicate rating
                continue
            global_ui_set.add((raw_uid, raw_iid))

            if not raw_uid in global_uid_map:
                unk_user_count += 1
            if not raw_iid in global_iid_map:
                unk_item_count += 1

            mapped_uid = global_uid_map.setdefault(raw_uid, len(global_uid_map))
            mapped_iid = global_iid_map.setdefault(raw_iid, len(global_iid_map))
            uid_map[raw_uid] = mapped_uid
            iid_map[raw_iid] = mapped_iid

            ur_list = user_ratings.setdefault(mapped_uid, [])
            ur_list.append((mapped_iid, float(rating)))

        if verbose:
            print('Number of tested users = {}'.format(len(user_ratings)))
            print('Number of unknown users = {}'.format(unk_user_count))
            print('Number of unknown items = {}'.format(unk_item_count))

        return self(user_ratings, uid_map, iid_map)


class MultimodalTestSet(TestSet):
    """Test Set

    Parameters
    ----------
    user_ratings: :obj:`defaultdict` of :obj:`list`
        The dictionary containing lists of tuples of the form (item, rating). The keys are user ids.

    uid_map: :obj:`defaultdict`
        The dictionary containing mapping from original ids to mapped ids of users.

    iid_map: :obj:`defaultdict`
        The dictionary containing mapping from original ids to mapped ids of items.

    """

    def __init__(self, user_ratings, uid_map, iid_map, **kwargs):
        super().__init__(user_ratings, uid_map, iid_map)
        self.add_modules(**kwargs)

    def add_modules(self, **kwargs):
        self.user_text = kwargs.get('user_text', None)
        self.item_text = kwargs.get('item_text', None)
        self.user_image = kwargs.get('user_image', None)
        self.item_image = kwargs.get('item_image', None)
        self.user_graph = kwargs.get('user_graph', None)
        self.item_graph = kwargs.get('item_graph', None)