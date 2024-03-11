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

import copy
import os
import pickle
import warnings
from collections import Counter, OrderedDict, defaultdict

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix

from ..utils import estimate_batches, get_rng, validate_format


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

        self.__user_ids = None
        self.__item_ids = None
        self.__user_data = None
        self.__item_data = None
        self.__chrono_user_data = None
        self.__chrono_item_data = None
        self.__csr_matrix = None
        self.__csc_matrix = None
        self.__dok_matrix = None

        self.ignored_attrs = [
            "__user_ids",
            "__item_ids",
            "__user_data",
            "__item_data",
            "__chrono_user_data",
            "__chrono_item_data",
            "__csr_matrix",
            "__csc_matrix",
            "__dok_matrix",
        ]

    @property
    def user_ids(self):
        """Return the list of raw user ids"""
        if self.__user_ids is None:
            self.__user_ids = list(self.uid_map.keys())
        return self.__user_ids

    @property
    def item_ids(self):
        """Return the list of raw item ids"""
        if self.__item_ids is None:
            self.__item_ids = list(self.iid_map.keys())
        return self.__item_ids

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
            self.__dok_matrix = dok_matrix((self.num_users, self.num_items), dtype="float")
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
            if exclude_unknowns and (uid not in global_uid_map or iid not in global_iid_map):
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
            np.asarray(u_indices, dtype="int"),
            np.asarray(i_indices, dtype="int"),
            np.asarray(r_values, dtype="float"),
        )

        timestamps = np.fromiter((int(data[i][3]) for i in valid_idx), dtype="int") if fmt == "UIRT" else None

        dataset = cls(
            num_users=len(global_uid_map),
            num_items=len(global_iid_map),
            uid_map=global_uid_map,
            iid_map=global_iid_map,
            uir_tuple=uir_tuple,
            timestamps=timestamps,
            seed=seed,
        )

        return dataset

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

    def num_user_batches(self, batch_size):
        """Estimate number of batches per epoch iterating over users"""
        return estimate_batches(self.num_users, batch_size)

    def num_item_batches(self, batch_size):
        """Estimate number of batches per epoch iterating over items"""
        return estimate_batches(self.num_items, batch_size)

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
                batch_ratings = np.concatenate((batch_ratings, np.zeros_like(neg_items)))

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
            raise ValueError("Unsupported negative sampling option: {}".format(neg_sampling))

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
        user_indices = np.fromiter(set(self.uir_tuple[0]), dtype="int")
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
        item_indices = np.fromiter(set(self.uir_tuple[1]), "int")
        for batch_ids in self.idx_iter(len(item_indices), batch_size, shuffle):
            yield item_indices[batch_ids]

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

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k in self.ignored_attrs:
                continue
            setattr(result, k, copy.deepcopy(v))
        return result

    def save(self, fpath):
        """Save a dataset to the filesystem.

        Parameters
        ----------
        fpath: str, required
            Path to a file for the dataset to be stored.

        """
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        dataset = copy.deepcopy(self)
        pickle.dump(dataset, open(fpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fpath):
        """Load a dataset from the filesystem.

        Parameters
        ----------
        fpath: str, required
            Path to a file where the dataset is stored.

        Returns
        -------
        self : object
        """
        dataset = pickle.load(open(fpath, "rb"))
        dataset.load_from = fpath  # for further loading
        return dataset


class BasketDataset(Dataset):
    """Training set contains history baskets

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

    basket_indices: numpy.array, required
        Array of basket indices corresponding to observation in `uir_tuple`.

    timestamps: numpy.array, optional, default: None
        Numpy array of timestamps corresponding to feedback in `uir_tuple`.
        This is only available when input data is in `UBIT` and `UBITJson` formats.

    extra_data: numpy.array, optional, default: None
        Array of json object corresponding to observations in `uir_tuple`.

    seed: int, optional, default: None
        Random seed for reproducing data sampling.

    Attributes
    ----------
    ubi_tuple: tuple
        Tuple (user_indices, baskets).

    timestamps: numpy.array
        Numpy array of timestamps corresponding to feedback in `ubi_tuple`.
        This is only available when input data is in `UTB` format.
    """

    def __init__(
        self,
        num_users,
        num_baskets,
        num_items,
        uid_map,
        bid_map,
        iid_map,
        uir_tuple,
        basket_indices=None,
        timestamps=None,
        extra_data=None,
        seed=None,
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            uid_map=uid_map,
            iid_map=iid_map,
            uir_tuple=uir_tuple,
            timestamps=timestamps,
            seed=seed,
        )
        self.num_baskets = num_baskets
        self.bid_map = bid_map
        self.basket_indices = basket_indices
        self.extra_data = extra_data
        basket_sizes = list(Counter(basket_indices).values())
        self.max_basket_size = np.max(basket_sizes)
        self.min_basket_size = np.min(basket_sizes)
        self.avg_basket_size = np.mean(basket_sizes)

        self.__baskets = None
        self.__basket_ids = None
        self.__user_basket_data = None
        self.__chrono_user_basket_data = None

    @property
    def basket_ids(self):
        """Return the list of raw basket ids"""
        if self.__basket_ids is None:
            self.__basket_ids = list(self.bid_map.keys())
        return self.__basket_ids

    @property
    def baskets(self):
        """A dictionary to store indices where basket ID appears in the data."""
        if self.__baskets is None:
            self.__baskets = defaultdict(list)
            for idx, bid in enumerate(self.basket_indices):
                self.__baskets[bid].append(idx)
        return self.__baskets

    @property
    def user_basket_data(self):
        """Data organized by user. A dictionary where keys are users,
        values are list of baskets purchased by corresponding users.
        """
        if self.__user_basket_data is None:
            self.__user_basket_data = defaultdict(list)
            for bid, ids in self.baskets.items():
                u = self.uir_tuple[0][ids[0]]
                self.__user_basket_data[u].append(bid)
        return self.__user_basket_data

    @property
    def chrono_user_basket_data(self):
        """Data organized by user sorted chronologically (timestamps required).
        A dictionary where keys are users, values are tuples of three chronologically
        sorted lists (baskets, timestamps) interacted by the corresponding users.
        """
        if self.__chrono_user_basket_data is None:
            assert self.timestamps is not None  # we need timestamps

            basket_timestamps = [self.timestamps[ids[0]] for ids in self.baskets.values()]  # one-off

            self.__chrono_user_basket_data = defaultdict(lambda: ([], []))
            for (bid, ids), t in zip(self.baskets.items(), basket_timestamps):
                u = self.uir_tuple[0][ids[0]]
                self.__chrono_user_basket_data[u][0].append(bid)
                self.__chrono_user_basket_data[u][1].append(t)

            # sorting based on timestamps
            for user, (baskets, timestamps) in self.__chrono_user_basket_data.items():
                sorted_idx = np.argsort(timestamps)
                sorted_baskets = [baskets[i] for i in sorted_idx]
                sorted_timestamps = [timestamps[i] for i in sorted_idx]
                self.__chrono_user_basket_data[user] = (
                    sorted_baskets,
                    sorted_timestamps,
                )

        return self.__chrono_user_basket_data

    @classmethod
    def build(
        cls,
        data,
        fmt="UBI",
        global_uid_map=None,
        global_bid_map=None,
        global_iid_map=None,
        seed=None,
        exclude_unknowns=False,
    ):
        """Constructing Dataset from given data of specific format.

        Parameters
        ----------
        data: list, required
            Data in the form of tuple (user, basket) for UB format,
            or tuple (user, timestamps, basket) for UTB format.

        fmt: str, default: 'UBI'
            Format of the input data. Currently, we are supporting:

            'UBI': User, Basket_ID, Item
            'UBIT': User, Basket_ID, Item, Timestamp
            'UBITJson': User, Basket_ID, Item, Timestamp, Extra data in Json format

        global_uid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of users.

        global_bid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of baskets.

        global_iid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of items.

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        exclude_unknowns: bool, default: False
            Ignore unknown users and items.

        Returns
        -------
        res: :obj:`<cornac.data.BasketDataset>`
            BasketDataset object.

        """
        fmt = validate_format(fmt, ["UBI", "UBIT", "UBITJson"])

        if global_uid_map is None:
            global_uid_map = OrderedDict()
        if global_bid_map is None:
            global_bid_map = OrderedDict()
        if global_iid_map is None:
            global_iid_map = OrderedDict()

        u_indices = []
        b_indices = []
        i_indices = []
        valid_idx = []
        extra_data = []
        for idx, (uid, bid, iid, *_) in enumerate(data):
            if exclude_unknowns and (iid not in global_iid_map):
                continue

            global_uid_map.setdefault(uid, len(global_uid_map))
            global_bid_map.setdefault(bid, len(global_bid_map))
            global_iid_map.setdefault(iid, len(global_iid_map))

            u_indices.append(global_uid_map[uid])
            b_indices.append(global_bid_map[bid])
            i_indices.append(global_iid_map[iid])
            valid_idx.append(idx)

        uir_tuple = (
            np.asarray(u_indices, dtype="int"),
            np.asarray(i_indices, dtype="int"),
            np.ones(len(u_indices), dtype="float"),
        )

        basket_indices = np.asarray(b_indices, dtype="int")

        timestamps = (
            np.fromiter((int(data[i][3]) for i in valid_idx), dtype="int") if fmt in ["UBIT", "UBITJson"] else None
        )

        extra_data = [data[i][4] for i in valid_idx] if fmt == "UBITJson" else None

        dataset = cls(
            num_users=len(global_uid_map),
            num_baskets=len(global_bid_map),
            num_items=len(global_iid_map),
            uid_map=global_uid_map,
            bid_map=global_bid_map,
            iid_map=global_iid_map,
            uir_tuple=uir_tuple,
            basket_indices=basket_indices,
            timestamps=timestamps,
            extra_data=extra_data,
            seed=seed,
        )

        return dataset

    @classmethod
    def from_ubi(cls, data, seed=None):
        """Constructing Dataset from UBI (User, Basket, Item) triples data.

        Parameters
        ----------
        data: list
            Data in the form of tuples (user, basket, item).

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.BasketDataset>`
            BasketDataset object.

        """
        return cls.build(data, fmt="UBI", seed=seed)

    @classmethod
    def from_ubit(cls, data, seed=None):
        """Constructing Dataset from UBIT format (User, Basket, Item, Timestamp)

        Parameters
        ----------
        data: tuple
            Data in the form of quadruples (user, basket, item, timestamp)

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.BasketDataset>`
            BasketDataset object.

        """
        return cls.build(data, fmt="UBIT", seed=seed)

    @classmethod
    def from_ubitjson(cls, data, seed=None):
        """Constructing Dataset from UBITJson format (User, Basket, Item, Timestamp, Json)

        Parameters
        ----------
        data: tuple
            Data in the form of tuples (user, basket, item, timestamp, json)

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.BasketDataset>`
            BasketDataset object.

        """
        return cls.build(data, fmt="UBITJson", seed=seed)

    def ub_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over data yielding batch of users and batch of baskets

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of users will be randomized. If `False`, default orders kept.

        Returns
        -------
        iterator : batch of user indices, batch of baskets corresponding to user indices

        """
        for batch_users in self.user_iter(batch_size, shuffle):
            batch_baskets = [self.user_basket_data[uid] for uid in batch_users]
            yield batch_users, batch_baskets

    def ubi_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over data yielding batch of users, basket ids, and batch of the corresponding items

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of users will be randomized. If `False`, default orders kept.

        Returns
        -------
        iterator : batch of user indices, batch of baskets corresponding to user indices, and batch of items correponding to baskets

        """
        _, item_indices, _ = self.uir_tuple
        for batch_users, batch_baskets in self.ub_iter(batch_size, shuffle):
            batch_basket_items = [
                [item_indices[self.baskets[bid]] for bid in user_baskets] for user_baskets in batch_baskets
            ]
            yield batch_users, batch_baskets, batch_basket_items

    def basket_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over data yielding batch of basket indices

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of triplets will be randomized. If `False`, default orders kept.

        Returns
        -------
        iterator : batch of basket indices (array of 'int')

        """
        basket_indices = np.fromiter(set(self.baskets.keys()), dtype="int")
        for batch_ids in self.idx_iter(len(basket_indices), batch_size, shuffle):
            yield basket_indices[batch_ids]


class SequentialDataset(Dataset):
    """Training set contains history sessions

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

    session_ids: numpy.array, required
        Array of session indices corresponding to observation in `uir_tuple`.

    timestamps: numpy.array, optional, default: None
        Numpy array of timestamps corresponding to feedback in `uir_tuple`.
        This is only available when input data is in `SIT`, `USIT`, SITJson`, and `USITJson` formats.

    extra_data: numpy.array, optional, default: None
        Array of json object corresponding to observations in `uir_tuple`.

    seed: int, optional, default: None
        Random seed for reproducing data sampling.

    Attributes
    ----------
    timestamps: numpy.array
        Numpy array of timestamps corresponding to feedback in `ubi_tuple`.
        This is only available when input data is in `UTB` format.
    """

    def __init__(
        self,
        num_users,
        num_sessions,
        num_items,
        uid_map,
        sid_map,
        iid_map,
        uir_tuple,
        session_indices=None,
        timestamps=None,
        extra_data=None,
        seed=None,
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            uid_map=uid_map,
            iid_map=iid_map,
            uir_tuple=uir_tuple,
            timestamps=timestamps,
            seed=seed,
        )
        self.num_sessions = num_sessions
        self.sid_map = sid_map
        self.session_indices = session_indices
        self.extra_data = extra_data
        session_sizes = list(Counter(session_indices).values())
        self.max_session_size = np.max(session_sizes)
        self.min_session_size = np.min(session_sizes)
        self.avg_session_size = np.mean(session_sizes)

        self.__sessions = None
        self.__session_ids = None
        self.__user_session_data = None
        self.__chrono_user_session_data = None

    @property
    def session_ids(self):
        """Return the list of raw session ids"""
        if self.__session_ids is None:
            self.__session_ids = list(self.sid_map.keys())
        return self.__session_ids

    @property
    def sessions(self):
        """A dictionary to store indices where session ID appears in the data."""
        if self.__sessions is None:
            self.__sessions = OrderedDict()
            for idx, sid in enumerate(self.session_indices):
                self.__sessions.setdefault(sid, [])
                self.__sessions[sid].append(idx)
        return self.__sessions

    @property
    def user_session_data(self):
        """Data organized by user. A dictionary where keys are users,
        values are list of sessions purchased by corresponding users.
        """
        if self.__user_session_data is None:
            self.__user_session_data = defaultdict(list)
            for sid, ids in self.sessions.items():
                u = self.uir_tuple[0][ids[0]]
                self.__user_session_data[u].append(sid)
        return self.__user_session_data

    @property
    def chrono_user_session_data(self):
        """Data organized by user sorted chronologically (timestamps required).
        A dictionary where keys are users, values are tuples of three chronologically
        sorted lists (sessions, timestamps) interacted by the corresponding users.
        """
        if self.__chrono_user_session_data is None:
            assert self.timestamps is not None  # we need timestamps

            session_timestamps = [self.timestamps[ids[0]] for ids in self.sessions.values()]  # one-off

            self.__chrono_user_session_data = defaultdict(lambda: ([], []))
            for (sid, ids), t in zip(self.sessions.items(), session_timestamps):
                u = self.uir_tuple[0][ids[0]]
                self.__chrono_user_session_data[u][0].append(sid)
                self.__chrono_user_session_data[u][1].append(t)

            # sorting based on timestamps
            for user, (sessions, timestamps) in self.__chrono_user_session_data.items():
                sorted_idx = np.argsort(timestamps)
                sorted_sessions = [sessions[i] for i in sorted_idx]
                sorted_timestamps = [timestamps[i] for i in sorted_idx]
                self.__chrono_user_session_data[user] = (
                    sorted_sessions,
                    sorted_timestamps,
                )

        return self.__chrono_user_session_data

    @classmethod
    def build(
        cls,
        data,
        fmt="SIT",
        global_uid_map=None,
        global_sid_map=None,
        global_iid_map=None,
        seed=None,
        exclude_unknowns=False,
    ):
        """Constructing Dataset from given data of specific format.

        Parameters
        ----------
        data: list, required
            Data in the form of tuple (user, session) for UB format,
            or tuple (user, timestamps, session) for UTB format.

        fmt: str, default: 'SIT'
            Format of the input data. Currently, we are supporting:

            'SIT': Session_ID, Item, Timestamp
            'USIT': User, Session_ID, Item, Timestamp
            'SITJson': Session_ID, Item, Timestamp, Extra data in Json format
            'USITJson': User, Session_ID, Item, Timestamp, Extra data in Json format

        global_uid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of users.

        global_sid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of sessions.

        global_iid_map: :obj:`defaultdict`, optional, default: None
            The dictionary containing global mapping from original ids to mapped ids of items.

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        exclude_unknowns: bool, default: False
            Ignore unknown users and items.

        Returns
        -------
        res: :obj:`<cornac.data.SequentialDataset>`
            SequentialDataset object.

        """
        fmt = validate_format(fmt, ["SIT", "USIT", "SITJson", "USITJson"])

        if global_uid_map is None:
            global_uid_map = OrderedDict()
        if global_sid_map is None:
            global_sid_map = OrderedDict()
        if global_iid_map is None:
            global_iid_map = OrderedDict()

        u_indices = []
        s_indices = []
        i_indices = []
        valid_idx = []
        extra_data = []
        for idx, tup in enumerate(data):
            uid, sid, iid, *_ = tup if fmt in ["USIT", "USITJson"] else [None] + list(tup)
            if exclude_unknowns and (iid not in global_iid_map):
                continue
            global_uid_map.setdefault(uid, len(global_uid_map))
            global_sid_map.setdefault(sid, len(global_sid_map))
            global_iid_map.setdefault(iid, len(global_iid_map))

            u_indices.append(global_uid_map[uid])
            s_indices.append(global_sid_map[sid])
            i_indices.append(global_iid_map[iid])
            valid_idx.append(idx)

        uir_tuple = (
            np.asarray(u_indices, dtype="int"),
            np.asarray(i_indices, dtype="int"),
            np.ones(len(u_indices), dtype="float"),
        )

        session_indices = np.asarray(s_indices, dtype="int")

        ts_pos = 3 if fmt in ["USIT", "USITJson"] else 2
        timestamps = (
            np.fromiter((int(data[i][ts_pos]) for i in valid_idx), dtype="int")
            if fmt in ["SIT", "SITJson", "USIT", "USITJson"]
            else None
        )

        extra_pos = ts_pos + 1
        extra_data = [data[i][extra_pos] for i in valid_idx] if fmt in ["SITJson", "USITJson"] else None

        dataset = cls(
            num_users=len(global_uid_map),
            num_sessions=len(set(session_indices)),
            num_items=len(global_iid_map),
            uid_map=global_uid_map,
            sid_map=global_sid_map,
            iid_map=global_iid_map,
            uir_tuple=uir_tuple,
            session_indices=session_indices,
            timestamps=timestamps,
            extra_data=extra_data,
            seed=seed,
        )

        return dataset

    @classmethod
    def from_sit(cls, data, seed=None):
        """Constructing Dataset from SIT (Session, Item, Timestamp) triples data.

        Parameters
        ----------
        data: list
            Data in the form of tuples (session, item, timestamp).

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.SequentialDataset>`
            SequentialDataset object.

        """
        return cls.build(data, fmt="SIT", seed=seed)

    @classmethod
    def from_usit(cls, data, seed=None):
        """Constructing Dataset from USIT format (User, Session, Item, Timestamp)

        Parameters
        ----------
        data: tuple
            Data in the form of quadruples (user, session, item, timestamp)

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.SequentialDataset>`
            SequentialDataset object.

        """
        return cls.build(data, fmt="USIT", seed=seed)

    @classmethod
    def from_sitjson(cls, data, seed=None):
        """Constructing Dataset from SITJson format (Session, Item, Timestamp, Json)

        Parameters
        ----------
        data: tuple
            Data in the form of tuples (session, item, timestamp, json)

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.SequentialDataset>`
            SequentialDataset object.

        """
        return cls.build(data, fmt="SITJson", seed=seed)

    @classmethod
    def from_usitjson(cls, data, seed=None):
        """Constructing Dataset from USITJson format (User, Session, Item, Timestamp, Json)

        Parameters
        ----------
        data: tuple
            Data in the form of tuples (user, session, item, timestamp, json)

        seed: int, optional, default: None
            Random seed for reproducing data sampling.

        Returns
        -------
        res: :obj:`<cornac.data.SequentialDataset>`
            SequentialDataset object.

        """
        return cls.build(data, fmt="USITJson", seed=seed)

    def num_batches(self, batch_size):
        """Estimate number of batches per epoch"""
        return estimate_batches(len(self.sessions), batch_size)

    def session_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over session indices

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of session_ids will be randomized. If `False`, default orders kept.

        Returns
        -------
        iterator : batch of session indices (array of 'int')

        """
        session_indices = np.array(list(self.sessions.keys()))
        for batch_ids in self.idx_iter(len(session_indices), batch_size, shuffle):
            batch_session_indices = session_indices[batch_ids]
            yield batch_session_indices

    def s_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over data yielding batch of sessions

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of sessions will be randomized. If `False`, default orders kept.

        Returns
        -------
        iterator : batch of session indices, batch of indices corresponding to session indices

        """
        for batch_session_ids in self.session_iter(batch_size, shuffle):
            batch_mapped_ids = [self.sessions[sid] for sid in batch_session_ids]
            yield batch_session_ids, batch_mapped_ids

    def si_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over data yielding batch of session indices, batch of mapped ids, and batch of sessions' items

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of triplets will be randomized. If `False`, default orders kept.

        Returns
        -------
        iterator : batch of session indices, batch mapped ids, batch of sessions' items (list of list)

        """
        for batch_session_indices, batch_mapped_ids in self.s_iter(batch_size, shuffle):
            batch_session_items = [[self.uir_tuple[1][i] for i in ids] for ids in batch_mapped_ids]
            yield batch_session_indices, batch_mapped_ids, batch_session_items

    def usi_iter(self, batch_size=1, shuffle=False):
        """Create an iterator over data yielding batch of user indices, batch of session indices, batch of mapped ids, and batch of sessions' items

        Parameters
        ----------
        batch_size: int, optional, default = 1

        shuffle: bool, optional, default: False
            If `True`, orders of triplets will be randomized. If `False`, default orders kept.

        Returns
        -------
        iterator : batch of user indices, batch of session indices (list of list), batch mapped ids (list of list of list), batch of sessions' items (list of list of list)

        """
        for user_indices in self.user_iter(batch_size, shuffle):
            batch_sids = [[sid for sid in self.user_session_data[uid]] for uid in user_indices]
            batch_mapped_ids = [[self.sessions[sid] for sid in self.user_session_data[uid]] for uid in user_indices]
            batch_session_items = [[[self.uir_tuple[1][i] for i in ids] for ids in u_batch_mapped_ids] for u_batch_mapped_ids in batch_mapped_ids]
            yield user_indices, batch_sids, batch_mapped_ids, batch_session_items
