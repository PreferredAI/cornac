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
import itertools
import warnings

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, dok_matrix

from ..utils import get_rng
from ..utils import validate_format
from ..utils import estimate_batches


class Dataset(object):
    """Training set contains preference matrix

    Parameters
    ----------
    num_users: int, required
        Number of users.

    num_items: int, required
        Number of items.

    uid_map: :obj:`OrderDict`, required
        The dictionary containing mapping from user original ids to mapped integer indices.

    iid_map: :obj:`OrderDict`, required
        The dictionary containing mapping from item original ids to mapped integer indices.

    uir_tuple: tuple, required
        Tuple of 3 numpy arrays (user_indices, item_indices, rating_values).

    timestamps: numpy.array, optional, default: None
        Array of timestamps corresponding to observations in `uir_tuple`.

    seed: int, optional, default: None
        Random seed for reproducing data sampling.

    Attributes
    ----------
    num_ratings: int
        Number of rating observations in the dataset.

    max_rating: float
        Maximum value among the rating observations.

    min_rating: float
        Minimum value among the rating observations.

    global_mean: float
        Average value over the rating observations.
        
    uir_tuple: tuple
        Tuple three numpy arrays (user_indices, item_indices, rating_values).

    timestamps: numpy.array
        Numpy array of timestamps corresponding to feedback in `uir_tuple`.
        This is only available when input data is in `UIRT` format.

    """

    def __init__(
        self,
        num_users,
        num_items,
        uid_map,
        iid_map,
        uir_tuple,
        timestamps=None,
        seed=None,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.uid_map = uid_map
        self.iid_map = iid_map
        self.uir_tuple = uir_tuple
        self.timestamps = timestamps
        self.seed = seed
        self.rng = get_rng(seed)

        (_, _, r_values) = uir_tuple
        self.num_ratings = len(r_values)
        self.max_rating = np.max(r_values)
        self.min_rating = np.min(r_values)
        self.global_mean = np.mean(r_values)

        self.__total_users = None
        self.__total_items = None
        self.__user_ids = None
        self.__item_ids = None
        self.__user_indices = None
        self.__item_indices = None

        self.__user_data = None
        self.__item_data = None
        self.__chrono_user_data = None
        self.__chrono_item_data = None
        self.__csr_matrix = None
        self.__csc_matrix = None
        self.__dok_matrix = None

    @property
    def total_users(self):
        """Total number of users including test and validation users if exists"""
        return self.__total_users if self.__total_users is not None else self.num_users

    @total_users.setter
    def total_users(self, input_value):
        """Set total number of users for the dataset"""
        assert input_value >= self.num_users
        self.__total_users = input_value

    @property
    def total_items(self):
        """Total number of items including test and validation items if exists"""
        return self.__total_items if self.__total_items is not None else self.num_items

    @total_items.setter
    def total_items(self, input_value):
        """Set total number of items for the dataset"""
        assert input_value >= self.num_items
        self.__total_items = input_value

    @property
    def user_ids(self):
        """An iterator over the raw user ids"""
        return self.uid_map.keys()

    @property
    def item_ids(self):
        """An iterator over the raw item ids"""
        return self.iid_map.keys()

    @property
    def user_indices(self):
        """An iterator over the user indices"""
        return self.uid_map.values()

    @property
    def item_indices(self):
        """An iterator over the item indices"""
        return self.iid_map.values()

    @property
    def user_data(self):
        """Data organized by user. A dictionary where keys are users,
        values are tuples of two lists (items, ratings) interacted by the corresponding users.
        """
        if self.__user_data is None:
            self.__user_data = defaultdict()
            for u, i, r in zip(*self.uir_tuple):
                u_data = self.__user_data.setdefault(u, ([], []))
                u_data[0].append(i)
                u_data[1].append(r)
        return self.__user_data

    @property
    def item_data(self):
        """Data organized by item. A dictionary where keys are items,
        values are tuples of two lists (users, ratings) interacted with the corresponding items.
        """
        if self.__item_data is None:
            self.__item_data = defaultdict()
            for u, i, r in zip(*self.uir_tuple):
                i_data = self.__item_data.setdefault(i, ([], []))
                i_data[0].append(u)
                i_data[1].append(r)
        return self.__item_data

    @property
    def chrono_user_data(self):
        """Data organized by user sorted chronologically (timestamps required).
        A dictionary where keys are users, values are tuples of three chronologically 
        sorted lists (items, ratings, timestamps) interacted by the corresponding users.
        """
        if self.timestamps is None:
            raise ValueError("Timestamps are required but None!")

        if self.__chrono_user_data is None:
            self.__chrono_user_data = defaultdict()
            for u, i, r, t in zip(*self.uir_tuple, self.timestamps):
                u_data = self.__chrono_user_data.setdefault(u, ([], [], []))
                u_data[0].append(i)
                u_data[1].append(r)
                u_data[2].append(t)
            # sorting based on timestamps
            for user, (items, ratings, timestamps) in self.__chrono_user_data.items():
                sorted_idx = np.argsort(timestamps)
                sorted_items = [items[i] for i in sorted_idx]
                sorted_ratings = [ratings[i] for i in sorted_idx]
                sorted_timestamps = [timestamps[i] for i in sorted_idx]
                self.__chrono_user_data[user] = (
                    sorted_items,
                    sorted_ratings,
                    sorted_timestamps,
                )
        return self.__chrono_user_data

    @property
    def chrono_item_data(self):
        """Data organized by item sorted chronologically (timestamps required).
        A dictionary where keys are items, values are tuples of three chronologically 
        sorted lists (users, ratings, timestamps) interacted with the corresponding items.
        """
        if self.timestamps is None:
            raise ValueError("Timestamps are required but None!")

        if self.__chrono_item_data is None:
            self.__chrono_item_data = defaultdict()
            for u, i, r, t in zip(*self.uir_tuple, self.timestamps):
                i_data = self.__chrono_item_data.setdefault(i, ([], [], []))
                i_data[0].append(u)
                i_data[1].append(r)
                i_data[2].append(t)
            # sorting based on timestamps
            for item, (users, ratings, timestamps) in self.__chrono_item_data.items():
                sorted_idx = np.argsort(timestamps)
                sorted_users = [users[i] for i in sorted_idx]
                sorted_ratings = [ratings[i] for i in sorted_idx]
                sorted_timestamps = [timestamps[i] for i in sorted_idx]
                self.__chrono_item_data[item] = (
                    sorted_users,
                    sorted_ratings,
                    sorted_timestamps,
                )
        return self.__chrono_item_data

    @property
    def matrix(self):
        """The user-item interaction matrix in CSR sparse format"""
        return self.csr_matrix

    @property
    def csr_matrix(self):
        """The user-item interaction matrix in CSR sparse format"""
        if self.__csr_matrix is None:
            (u_indices, i_indices, r_values) = self.uir_tuple
            self.__csr_matrix = csr_matrix(
                (r_values, (u_indices, i_indices)),
                shape=(self.num_users, self.num_items),
            )
        return self.__csr_matrix

    @property
    def csc_matrix(self):
        """The user-item interaction matrix in CSC sparse format"""
        if self.__csc_matrix is None:
            (u_indices, i_indices, r_values) = self.uir_tuple
            self.__csc_matrix = csc_matrix(
                (r_values, (u_indices, i_indices)),
                shape=(self.num_users, self.num_items),
            )
        return self.__csc_matrix

    @property
    def dok_matrix(self):
        """The user-item interaction matrix in DOK sparse format"""
        if self.__dok_matrix is None:
            self.__dok_matrix = dok_matrix(
                (self.num_users, self.num_items), dtype='float'
            )
            for u, i, r in zip(*self.uir_tuple):
                self.__dok_matrix[u, i] = r
        return self.__dok_matrix

    @classmethod
    def build(
        cls,
        data,
        fmt="UIR",
        global_uid_map=None,
        global_iid_map=None,
        seed=None,
        exclude_unknowns=False,
    ):
        """Constructing Dataset from given data of specific format.

        Parameters
        ----------
        data: array-like, required
            Data in the form of triplets (user, item, rating) for UIR format,
            or quadruplets (user, item, rating, timestamps) for UIRT format.

        fmt: str, default: 'UIR'
            Format of the input data. Currently, we are supporting:

            'UIR': User, Item, Rating
            'UIRT': User, Item, Rating, Timestamp

        global_uid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of users.

        global_iid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of items.

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        exclude_unknowns: bool, default: False
            Ignore unknown users and items.

        Returns
        -------
        res: :obj:`<cornac.data.Dataset>`
            Dataset object.

        """
        fmt = validate_format(fmt, ["UIR", "UIRT"])

        if global_uid_map is None:
            global_uid_map = OrderedDict()
        if global_iid_map is None:
            global_iid_map = OrderedDict()

        uid_map = OrderedDict()
        iid_map = OrderedDict()

        u_indices = []
        i_indices = []
        r_values = []
        valid_idx = []

        ui_set = set()  # avoid duplicate observations
        dup_count = 0

        for idx, (uid, iid, rating, *_) in enumerate(data):
            if exclude_unknowns and (
                uid not in global_uid_map or iid not in global_iid_map
            ):
                continue

            if (uid, iid) in ui_set:
                dup_count += 1
                continue
            ui_set.add((uid, iid))

            uid_map[uid] = global_uid_map.setdefault(uid, len(global_uid_map))
            iid_map[iid] = global_iid_map.setdefault(iid, len(global_iid_map))

            u_indices.append(uid_map[uid])
            i_indices.append(iid_map[iid])
            r_values.append(float(rating))
            valid_idx.append(idx)

        if dup_count > 0:
            warnings.warn("%d duplicated observations are removed!" % dup_count)

        if len(ui_set) == 0:
            raise ValueError("data is empty after being filtered!")

        uir_tuple = (
            np.asarray(u_indices, dtype='int'),
            np.asarray(i_indices, dtype='int'),
            np.asarray(r_values, dtype='float'),
        )

        timestamps = (
            np.fromiter((int(data[i][3]) for i in valid_idx), dtype='int')
            if fmt == "UIRT"
            else None
        )

        return cls(
            num_users=len(global_uid_map),
            num_items=len(global_iid_map),
            uid_map=uid_map,
            iid_map=iid_map,
            uir_tuple=uir_tuple,
            timestamps=timestamps,
            seed=seed,
        )

    @classmethod
    def from_uir(cls, data, seed=None):
        """Constructing Dataset from UIR (User, Item, Rating) triplet data.

        Parameters
        ----------
        data: array-like, shape: [n_examples, 3]
            Data in the form of triplets (user, item, rating)

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.Dataset>`
            Dataset object.

        """
        return cls.build(data, fmt="UIR", seed=seed)

    @classmethod
    def from_uirt(cls, data, seed=None):
        """Constructing Dataset from UIRT (User, Item, Rating, Timestamp) 
        quadruplet data.

        Parameters
        ----------
        data: array-like, shape: [n_examples, 4]
            Data in the form of triplets (user, item, rating, timestamp)

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.Dataset>`
            Dataset object.

        """
        return cls.build(data, fmt="UIRT", seed=seed)

    def reset(self):
        """Reset the random number generator for reproducibility"""
        self.rng = get_rng(self.seed)
        return self

    def num_batches(self, batch_size):
        """Estimate number of batches per epoch"""
        return estimate_batches(len(self.uir_tuple[0]), batch_size)

    def idx_iter(self, idx_range, batch_size=1, shuffle=False):
        """Create an iterator over batch of indices

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of indices (array of 'int')

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

    def uir_iter(self, batch_size=1, shuffle=False, binary=False, num_zeros=0):
        """Create an iterator over data yielding batch of users, items, and rating values

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of triplets will be randomized. If `False`, default orders kept.

        binary: bool, optional, default: False
            If `True`, non-zero ratings will be turned into `1`, otherwise, values remain unchanged.

        num_zeros: int, optional, default = 0
            Number of unobserved ratings (zeros) to be added per user. This could be used
            for negative sampling. By default, no values are added.

        Returns
        -------
        iterator : batch of users (array of 'int'), batch of items (array of 'int'),
            batch of ratings (array of 'float')

        """
        for batch_ids in self.idx_iter(len(self.uir_tuple[0]), batch_size, shuffle):
            batch_users = self.uir_tuple[0][batch_ids]
            batch_items = self.uir_tuple[1][batch_ids]
            if binary:
                batch_ratings = np.ones_like(batch_items)
            else:
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
                batch_ratings = np.concatenate(
                    (batch_ratings, np.zeros_like(neg_items))
                )

            yield batch_users, batch_items, batch_ratings

    def uij_iter(self, batch_size=1, shuffle=False, neg_sampling="uniform"):
        """Create an iterator over data yielding batch of users, positive items, and negative items

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of triplets will be randomized. If `False`, default orders kept.

        neg_sampling: str, optional, default: 'uniform'
            How negative item `j` will be sampled. Supported options: {`uniform`, `popularity`}.

        Returns
        -------
        iterator : batch of users (array of 'int'), batch of positive items (array of 'int'),
            batch of negative items (array of 'int')

        """

        if neg_sampling.lower() == "uniform":
            neg_population = np.arange(self.num_items)
        elif neg_sampling.lower() == "popularity":
            neg_population = self.uir_tuple[1]
        else:
            raise ValueError(
                "Unsupported negative sampling option: {}".format(neg_sampling)
            )

        for batch_ids in self.idx_iter(len(self.uir_tuple[0]), batch_size, shuffle):
            batch_users = self.uir_tuple[0][batch_ids]
            batch_pos_items = self.uir_tuple[1][batch_ids]
            batch_pos_ratings = self.uir_tuple[2][batch_ids]
            batch_neg_items = np.empty_like(batch_pos_items)
            for i, (user, pos_rating) in enumerate(zip(batch_users, batch_pos_ratings)):
                neg_item = self.rng.choice(neg_population)
                while self.dok_matrix[user, neg_item] >= pos_rating:
                    neg_item = self.rng.choice(neg_population)
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
        iterator : batch of user indices (array of 'int')
        """
        user_indices = np.fromiter(self.user_indices, dtype='int')
        for batch_ids in self.idx_iter(len(user_indices), batch_size, shuffle):
            yield user_indices[batch_ids]

    def item_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over item indices

        Parameters
        ----------
        batch_size : int, optional, default = 1

        shuffle : bool, optional
            If True, orders of triplets will be randomized. If False, default orders kept

        Returns
        -------
        iterator : batch of item indices (array of 'int')
        """
        item_indices = np.fromiter(self.item_indices, 'int')
        for batch_ids in self.idx_iter(len(item_indices), batch_size, shuffle):
            yield item_indices[batch_ids]

    def is_unk_user(self, user_idx):
        """Return whether or not a user is unknown given the user index"""
        return user_idx >= self.num_users

    def is_unk_item(self, item_idx):
        """Return whether or not an item is unknown given the item index"""
        return item_idx >= self.num_items

    def add_modalities(self, **kwargs):
        self.user_feature = kwargs.get("user_feature", None)
        self.item_feature = kwargs.get("item_feature", None)
        self.user_text = kwargs.get("user_text", None)
        self.item_text = kwargs.get("item_text", None)
        self.user_image = kwargs.get("user_image", None)
        self.item_image = kwargs.get("item_image", None)
        self.user_graph = kwargs.get("user_graph", None)
        self.item_graph = kwargs.get("item_graph", None)
        self.sentiment = kwargs.get("sentiment", None)
        self.review_text = kwargs.get("review_text", None)

