# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from scipy.sparse import csr_matrix
from collections import OrderedDict
from ..utils import estimate_batches
import numpy as np


class TrainSet:
    """Training Set

    Parameters
    ----------
    uid_map: :obj:`defaultdict`
        The dictionary containing mapping from original ids to mapped ids of users.

    iid_map: :obj:`defaultdict`
        The dictionary containing mapping from original ids to mapped ids of items.

    """

    def __init__(self, uid_map, iid_map):
        self._uid_map = uid_map
        self._iid_map = iid_map

    @property
    def num_users(self):
        """Return the number of users"""
        return len(self._uid_map)

    @property
    def num_items(self):
        """Return the number of items"""
        return len(self._iid_map)

    @property
    def uid_list(self):
        """Return the list of mapped user ids"""
        return list(self._uid_map.values())

    @property
    def raw_uid_list(self):
        """Return the list of raw user ids"""
        return list(self._uid_map.keys())

    @property
    def iid_list(self):
        """Return the list of mapped item ids"""
        return list(self._iid_map.values())

    @property
    def raw_iid_list(self):
        """Return the list of raw item ids"""
        return list(self._iid_map.keys())

    def is_unk_user(self, mapped_uid):
        """Return whether or not a user is unknown given the mapped id"""
        return mapped_uid >= self.num_users

    def is_unk_item(self, mapped_iid):
        """Return whether or not an item is unknown given the mapped id"""
        return mapped_iid >= self.num_items

    def get_uid(self, raw_uid):
        """Return the mapped id of a user given a raw id"""
        return self._uid_map[raw_uid]

    def get_iid(self, raw_iid):
        """Return the mapped id of an item given a raw id"""
        return self._iid_map[raw_iid]

    @staticmethod
    def idx_iter(idx_range, batch_size=1, shuffle=False):
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
            np.random.shuffle(indices)

        n_batches = estimate_batches(len(indices), batch_size)
        for b in range(n_batches):
            start_offset = batch_size * b
            end_offset = batch_size * b + batch_size
            end_offset = min(end_offset, len(indices))
            batch_ids = indices[start_offset:end_offset]
            yield batch_ids


class MatrixTrainSet(TrainSet):
    """Training set contains preference matrix

    Parameters
    ----------
    matrix: :obj:`scipy.sparse.csr_matrix`
        Preferences in the form of scipy sparse matrix.

    max_rating: float
        Maximum value of the preferences.

    min_rating: float
        Minimum value of the preferences.

    global_mean: float
        Average value of the preferences.

    uid_map: :obj:`defaultdict`
        The dictionary containing mapping from original ids to mapped ids of users.

    iid_map: :obj:`defaultdict`
        The dictionary containing mapping from original ids to mapped ids of items.

    """

    def __init__(self, matrix, max_rating, min_rating, global_mean, uid_map, iid_map):
        TrainSet.__init__(self, uid_map, iid_map)
        self.matrix = matrix
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.global_mean = global_mean
        self.item_ppl_rank, self.item_ppl_scores = self._rank_items_by_popularity(matrix)
        self.uir_tuple = None

    @property
    def uir_tuple(self):
        if not self.__uir_tuple:
            # rating matrix is assumed in the CSR format
            if not self.matrix.has_sorted_indices:
                self.matrix.sort_indices()

            # this basically calculates the 'row' attribute of a COO matrix
            # without requiring us to get the whole COO matrix
            num_users = self.matrix.shape[0]
            user_counts = np.ediff1d(self.matrix.indptr)
            user_ids = np.repeat(np.arange(num_users), user_counts).astype(self.matrix.indices.dtype)

            self.__uir_tuple = (user_ids, self.matrix.indices, self.matrix.data)
        return self.__uir_tuple

    @uir_tuple.setter
    def uir_tuple(self, input_tuple):
        if input_tuple is not None and len(input_tuple) != 3:
            raise ValueError('input_tuple required to be size 3 but size {}'.format(len(input_tuple)))
        self.__uir_tuple = input_tuple

    @staticmethod
    def _rank_items_by_popularity(rating_matrix):
        """Rank items by their popularity.

        Parameters
        ----------
        rating_matrix: :obj:`scipy.sparse.csr_matrix`
            Preference matrix.

        Returns
        -------
        train_set: :obj:`<numpy.matrix>`
        Ranked matrix according to item popularity.

        """
        item_scores = rating_matrix.sum(axis=0)
        item_rank = np.argsort(item_scores.A1)[::-1]
        return item_rank, item_scores

    @classmethod
    def from_uir(cls, data, global_uid_map=None, global_iid_map=None,
                 global_ui_set=None, verbose=False):
        """Constructing TrainSet from triplet data.

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

        verbose: bool, default: False
            The verbosity flag.

        Returns
        -------
        train_set: :obj:`<cornac.data.MatrixTrainSet>`
            MatrixTrainSet object.

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

        for raw_uid, raw_iid, rating in data:
            if (raw_uid, raw_iid) in global_ui_set:  # duplicate rating
                continue
            global_ui_set.add((raw_uid, raw_iid))

            mapped_uid = global_uid_map.setdefault(raw_uid, len(global_uid_map))
            mapped_iid = global_iid_map.setdefault(raw_iid, len(global_iid_map))
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

        train_set = cls(csr_mat, max_rating, min_rating, global_mean, uid_map, iid_map)

        # since we have triplet arrays, let's construct uir_tuple for the train_set
        train_set.uir_tuple = (np.asarray(u_indices, dtype=np.int),
                               np.asarray(i_indices, dtype=np.int),
                               np.asarray(r_values, dtype=np.float))

        return train_set

    def num_batches(self, batch_size):
        return estimate_batches(len(self.uir_tuple[0]), batch_size)

    def uir_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over data yielding batch of users, items, and rating values

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
        for batch_ids in self.idx_iter(len(self.uir_tuple[0]), batch_size, shuffle):
            batch_users = self.uir_tuple[0][batch_ids]
            batch_items = self.uir_tuple[1][batch_ids]
            batch_ratings = self.uir_tuple[2][batch_ids]
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
                neg_item = np.random.randint(0, self.num_items)
                while self.matrix[user, neg_item] >= pos_rating:
                    neg_item = np.random.randint(0, self.num_items)
                batch_neg_items[i] = neg_item
            yield batch_users, batch_pos_items, batch_neg_items

    def user_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over user ids

        Parameters
        ----------
        batch_size : int, optional, default = 1

        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of user ids (array of np.int)
        """
        user_ids = np.arange(self.num_users)
        for batch_ids in self.idx_iter(self.num_users, batch_size, shuffle):
            yield user_ids[batch_ids]

    def item_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over item ids

        Parameters
        ----------
        batch_size : int, optional, default = 1

        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of item ids (array of np.int)
        """
        item_ids = np.arange(self.num_items)
        for batch_ids in self.idx_iter(self.num_items, batch_size, shuffle):
            yield item_ids[batch_ids]


class MultimodalTrainSet(MatrixTrainSet):
    """Multimodal training set

    Parameters
    ----------
    matrix: :obj:`scipy.sparse.csr_matrix`
        Preferences in the form of scipy sparse matrix.

    max_rating: float
        Maximum value of the preferences.

    min_rating: float
        Minimum value of the preferences.

    global_mean: float
        Average value of the preferences.

    uid_map: :obj:`defaultdict`
        The dictionary containing mapping from original ids to mapped ids of users.

    iid_map: :obj:`defaultdict`
        The dictionary containing mapping from original ids to mapped ids of items.

    """

    def __init__(self, matrix, max_rating, min_rating, global_mean, uid_map, iid_map, **kwargs):
        super().__init__(matrix, max_rating, min_rating, global_mean, uid_map, iid_map)
        self.add_modules(**kwargs)

    def add_modules(self, **kwargs):
        self.user_text = kwargs.get('user_text', None)
        self.item_text = kwargs.get('item_text', None)
        self.user_image = kwargs.get('user_image', None)
        self.item_image = kwargs.get('item_image', None)
        self.user_graph = kwargs.get('user_graph', None)
        self.item_graph = kwargs.get('item_graph', None)
