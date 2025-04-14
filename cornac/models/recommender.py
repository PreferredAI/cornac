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
import inspect
import os
import pickle
import warnings
from datetime import datetime
from glob import glob
import json

import numpy as np

from ..exception import ScoreException
from ..utils.common import clip

MEASURE_L2 = "l2 distance aka. Euclidean distance"
MEASURE_DOT = "dot product aka. inner product"
MEASURE_COSINE = "cosine similarity"


def is_ann_supported(recom):
    """Return True if the given recommender model support ANN search.

    Parameters
    ----------
    recom : recommender model
        Recommender object to test.

    Returns
    -------
    out : bool
        True if recom supports ANN search and False otherwise.
    """
    return getattr(recom, "_ann_supported", False)


class ANNMixin:
    """Mixin class for Approximate Nearest Neighbor Search."""

    _ann_supported = True

    def get_vector_measure(self):
        """Getting a valid choice of vector measurement in ANNMixin._measures.

        Returns
        -------
        :raise NotImplementedError
        """
        raise NotImplementedError()

    def get_user_vectors(self):
        """Getting a matrix of user vectors serving as query for ANN search.

        Returns
        -------
        :raise NotImplementedError
        """
        raise NotImplementedError()

    def get_item_vectors(self):
        """Getting a matrix of item vectors used for building the index for ANN search.

        Returns
        -------
        :raise NotImplementedError
        """
        raise NotImplementedError()


class Recommender:
    """Generic class for a recommender model. All recommendation models should inherit from this class.

    Parameters
    ----------------
    name: str, required
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trainable.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    Attributes
    ----------
    num_users: int
        Number of users in training data.

    num_items: int
        Number of items in training data.

    total_users: int
        Number of users in training, validation, and test data.
        In other words, this includes unknown/unseen users.

    total_items: int
        Number of items in training, validation, and test data.
        In other words, this includes unknown/unseen items.

    uid_map: int
        Global mapping of user ID-index.

    iid_map: int
        Global mapping of item ID-index.

    max_rating: float
        Maximum value among the rating observations.

    min_rating: float
        Minimum value among the rating observations.

    global_mean: float
        Average value over the rating observations.
    """

    def __init__(self, name, trainable=True, verbose=False):
        self.name = name
        self.trainable = trainable
        self.verbose = verbose
        self.is_fitted = False

        # attributes to be ignored when saving model
        self.ignored_attrs = ["train_set", "val_set", "test_set"]

        # useful information getting from train_set for prediction
        self.num_users = None
        self.num_items = None
        self.uid_map = None
        self.iid_map = None
        self.max_rating = None
        self.min_rating = None
        self.global_mean = None

        self.__user_ids = None
        self.__item_ids = None

    @property
    def total_users(self):
        """Total number of users including users in test and validation if exists"""
        return len(self.uid_map) if self.uid_map is not None else self.num_users

    @property
    def total_items(self):
        """Total number of items including users in test and validation if exists"""
        return len(self.iid_map) if self.iid_map is not None else self.num_items

    @property
    def user_ids(self):
        """Return the list of raw user IDs"""
        if self.__user_ids is None:
            self.__user_ids = list(self.uid_map.keys())
        return self.__user_ids

    @property
    def item_ids(self):
        """Return the list of raw item IDs"""
        if self.__item_ids is None:
            self.__item_ids = list(self.iid_map.keys())
        return self.__item_ids

    def reset_info(self):
        self.best_value = float("-inf")
        self.best_epoch = 0
        self.current_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        ignored_attrs = set(self.ignored_attrs)
        for k, v in self.__dict__.items():
            if k in ignored_attrs:
                continue
            setattr(result, k, copy.deepcopy(v))
        return result

    @classmethod
    def _get_init_params(cls):
        """Get initial parameters from the model constructor"""
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != "self"]

        return sorted([p.name for p in parameters])

    def clone(self, new_params=None):
        """Clone an instance of the model object.

        Parameters
        ----------
        new_params: dict, optional, default: None
            New parameters for the cloned instance.

        Returns
        -------
        object: :obj:`cornac.models.Recommender`
        """
        new_params = {} if new_params is None else new_params
        init_params = {}
        for name in self._get_init_params():
            init_params[name] = new_params.get(name, copy.deepcopy(getattr(self, name)))

        return self.__class__(**init_params)

    def save(self, save_dir=None, save_trainset=False, metadata=None):
        """Save a recommender model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

        save_trainset: bool, default: False
            Save train_set together with the model. This is useful
            if we want to deploy model later because train_set is
            required for certain evaluation steps.

        metadata: dict, default: None
            Metadata to be saved with the model. This is useful
            to store model details.

        Returns
        -------
        model_file : str
            Path to the model file stored on the filesystem.
        """
        if save_dir is None:
            return

        model_dir = os.path.join(save_dir, self.name)
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        model_file = os.path.join(model_dir, "{}.pkl".format(timestamp))

        saved_model = copy.deepcopy(self)
        pickle.dump(
            saved_model, open(model_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )
        if self.verbose:
            print("{} model is saved to {}".format(self.name, model_file))

        metadata = {} if metadata is None else metadata
        metadata["model_classname"] = type(saved_model).__name__
        metadata["model_file"] = os.path.basename(model_file)

        if save_trainset:
            trainset_file = model_file + ".trainset"
            pickle.dump(
                self.train_set,
                open(trainset_file, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            metadata["trainset_file"] = os.path.basename(trainset_file)

        with open(model_file + ".meta", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem.

        Parameters
        ----------
        model_path: str, required
            Path to a file or directory where the model is stored. If a directory is
            provided, the latest model will be loaded.

        trainable: boolean, optional, default: False
            Set it to True if you would like to finetune the model. By default,
            the model parameters are assumed to be fixed after being loaded.

        Returns
        -------
        self : object
        """
        if os.path.isdir(model_path):
            model_file = sorted(glob("{}/*.pkl".format(model_path)))[-1]
        else:
            model_file = model_path

        model = pickle.load(open(model_file, "rb"))
        model.trainable = trainable
        model.load_from = model_file  # for further loading
        return model

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        if self.is_fitted:
            warnings.warn(
                "Model is already fitted. Re-fitting will overwrite the previous model."
            )

        self.reset_info()
        train_set.reset()
        if val_set is not None:
            val_set.reset()

        # get some useful information for prediction
        self.num_users = train_set.num_users
        self.num_items = train_set.num_items
        self.uid_map = train_set.uid_map
        self.iid_map = train_set.iid_map
        self.min_rating = train_set.min_rating
        self.max_rating = train_set.max_rating
        self.global_mean = train_set.global_mean

        # just for future wrapper to call fit(), not supposed to be used during prediction
        self.train_set = train_set
        self.val_set = val_set

        self.is_fitted = True

        return self

    def knows_user(self, user_idx):
        """Return whether the model knows user by its index

        Parameters
        ----------
        user_idx: int, required
            The index of the user (not the original user ID).

        Returns
        -------
        res : bool
            True if model knows the user from traning data, False otherwise.
        """
        return user_idx is not None and user_idx >= 0 and user_idx < self.num_users

    def knows_item(self, item_idx):
        """Return whether the model knows item by its index

        Parameters
        ----------
        item_idx: int, required
            The index of the item (not the original item ID).

        Returns
        -------
        res : bool
            True if model knows the item from traning data, False otherwise.
        """
        return item_idx is not None and item_idx >= 0 and item_idx < self.num_items

    def is_unknown_user(self, user_idx):
        """Return whether the model knows user by its index. Reverse of knows_user() function,
        for better readability in some cases.

        Parameters
        ----------
        user_idx: int, required
            The index of the user (not the original user ID).

        Returns
        -------
        res : bool
            True if model knows the user from traning data, False otherwise.
        """
        return not self.knows_user(user_idx)

    def is_unknown_item(self, item_idx):
        """Return whether the model knows item by its index. Reverse of knows_item() function,
        for better readability in some cases.

        Parameters
        ----------
        item_idx: int, required
            The index of the item (not the original item ID).

        Returns
        -------
        res : bool
            True if model knows the item from traning data, False otherwise.
        """
        return not self.knows_item(item_idx)

    def transform(self, test_set):
        """Transform test set into cached results accelerating the score function.
        This function is supposed to be called in the `cornac.eval_methods.BaseMethod`
        before evaluation step. It is optional for this function to be implemented.

        Parameters
        ----------
        test_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        """
        pass

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        raise NotImplementedError("The algorithm is not able to make score prediction!")

    def default_score(self):
        """Overwrite this function if your algorithm has special treatment for cold-start problem"""
        return self.global_mean

    def rate(self, user_idx, item_idx, clipping=True):
        """Give a rating score between pair of user and item

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform item raking.

        item_idx: int, required
            The index of the item to be rated by the user.

        clipping: bool, default: True
            Whether to clip the predicted rating value.

        Returns
        -------
        A scalar
            A rating score of the user for the item
        """
        try:
            rating_pred = self.score(user_idx, item_idx)
        except ScoreException:
            rating_pred = self.default_score()

        if clipping:
            rating_pred = clip(rating_pred, self.min_rating, self.max_rating)

        return rating_pred

    def rank(self, user_idx, item_indices=None, k=-1, **kwargs):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform item raking.

        item_indices: 1d array, optional, default: None
            A list of candidate item indices to be ranked by the user.
            If `None`, list of ranked known item indices and their scores will be returned.

        k: int, required
            Cut-off length for recommendations, k=-1 will return ranked list of all items.
            This is more important for ANN to know the limit to avoid exhaustive ranking.

        Returns
        -------
        (ranked_items, item_scores): tuple
            `ranked_items` contains item indices being ranked by their scores.
            `item_scores` contains scores of items corresponding to index in `item_indices` input.

        """
        # obtain item scores from the model
        try:
            known_item_scores = self.score(user_idx, **kwargs)
        except ScoreException:
            known_item_scores = np.ones(self.total_items) * self.default_score()

        # check if the returned scores also cover unknown items
        # if not, all unknown items will be given the MIN score
        if len(known_item_scores) == self.total_items:
            all_item_scores = known_item_scores
        else:
            all_item_scores = np.ones(self.total_items) * np.min(known_item_scores)
            all_item_scores[: self.num_items] = known_item_scores

        # rank items based on their scores
        item_indices = (
            np.arange(self.num_items)
            if item_indices is None
            else np.asarray(item_indices)
        )
        item_scores = all_item_scores[item_indices]

        if k != -1:  # O(n + k log k), faster for small k which is usually the case
            partitioned_idx = np.argpartition(item_scores, -k)
            top_k_idx = partitioned_idx[-k:]
            sorted_top_k_idx = top_k_idx[np.argsort(item_scores[top_k_idx])]
            partitioned_idx[-k:] = sorted_top_k_idx
            ranked_items = item_indices[partitioned_idx[::-1]]
        else:  # O(n log n)
            ranked_items = item_indices[item_scores.argsort()[::-1]]

        return ranked_items, item_scores

    def recommend(self, user_id, k=-1, remove_seen=False, train_set=None):
        """Generate top-K item recommendations for a given user. Key difference between
        this function and rank() function is that rank() function works with mapped
        user/item index while this function works with original user/item ID. This helps
        hide the abstraction of ID-index mapping, and make model usage and deployment cleaner.

        Parameters
        ----------
        user_id: str, required
            The original ID of the user.

        k: int, optional, default=-1
            Cut-off length for recommendations, k=-1 will return ranked list of all items.

        remove_seen: bool, optional, default: False
            Remove seen/known items during training and validation from output recommendations.

        train_set: :obj:`cornac.data.Dataset`, optional, default: None
            Training dataset needs to be provided in order to remove seen items.

        Returns
        -------
        recommendations: list
            Recommended items in the form of their original IDs.
        """
        user_idx = self.uid_map.get(user_id, -1)
        if user_idx == -1:
            raise ValueError(f"{user_id} is unknown to the model.")

        if k < -1 or k > self.total_items:
            raise ValueError(
                f"k={k} is invalid, there are {self.total_users} users in total."
            )

        item_indices = np.arange(self.total_items)
        if remove_seen:
            seen_mask = np.zeros(len(item_indices), dtype="bool")
            if train_set is None:
                raise ValueError("train_set must be provided to remove seen items.")
            if user_idx < train_set.csr_matrix.shape[0]:
                seen_mask[train_set.csr_matrix.getrow(user_idx).indices] = True
                item_indices = item_indices[~seen_mask]

        item_rank, _ = self.rank(user_idx, item_indices)
        if k != -1:
            item_rank = item_rank[:k]

        recommendations = [self.item_ids[i] for i in item_rank]
        return recommendations

    def monitor_value(self, train_set, val_set):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.
        Note: `val_set` could be `None` thus it needs to be checked before usage.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        :raise NotImplementedError
        """
        raise NotImplementedError()

    def early_stop(self, train_set, val_set, min_delta=0.0, patience=0):
        """Check if training should be stopped when validation loss has stopped improving.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        min_delta: float, optional, default: 0.
            The minimum increase in monitored value on validation set to be considered as improvement,
            i.e. an increment of less than `min_delta` will count as no improvement.

        patience: int, optional, default: 0
            Number of epochs with no improvement after which training should be stopped.

        Returns
        -------
        res : bool
            Return `True` if model training should be stopped (no improvement on validation set),
            otherwise return `False`.
        """
        self.current_epoch += 1
        current_value = self.monitor_value(train_set, val_set)
        if current_value is None:
            return False

        if np.greater_equal(current_value - self.best_value, min_delta):
            self.best_value = current_value
            self.best_epoch = self.current_epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= patience:
                self.stopped_epoch = self.current_epoch

        if self.stopped_epoch > 0:
            print("Early stopping:")
            print(
                "- best epoch = {}, stopped epoch = {}".format(
                    self.best_epoch, self.stopped_epoch
                )
            )
            print(
                "- best monitored value = {:.6f} (delta = {:.6f})".format(
                    self.best_value, current_value - self.best_value
                )
            )
            return True
        return False


class NextBasketRecommender(Recommender):
    """Generic class for a next basket recommender model. All next basket recommendation models should inherit from this class.

    Parameters
    ----------------
    name: str, required
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trainable.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    Attributes
    ----------
    num_users: int
        Number of users in training data.

    num_items: int
        Number of items in training data.

    total_users: int
        Number of users in training, validation, and test data.
        In other words, this includes unknown/unseen users.

    total_items: int
        Number of items in training, validation, and test data.
        In other words, this includes unknown/unseen items.

    uid_map: int
        Global mapping of user ID-index.

    iid_map: int
        Global mapping of item ID-index.
    """

    def __init__(self, name, trainable=True, verbose=False):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

    def score(self, user_idx, history_baskets, **kwargs):
        """Predict the scores for all items based on input history baskets

        Parameters
        ----------
        history_baskets: list of lists
            The list of history baskets in sequential manner for next-basket prediction.

        Returns
        -------
        res : a Numpy array
            Relative scores of all known items

        """
        raise NotImplementedError("The algorithm is not able to make score prediction!")


class NextItemRecommender(Recommender):
    """Generic class for a next item recommender model. All next item recommendation models should inherit from this class.

    Parameters
    ----------------
    name: str, required
        Name of the recommender model.

    trainable: boolean, optional, default: True
        When False, the model is not trainable.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    Attributes
    ----------
    num_users: int
        Number of users in training data.

    num_items: int
        Number of items in training data.

    total_users: int
        Number of users in training, validation, and test data.
        In other words, this includes unknown/unseen users.

    total_items: int
        Number of items in training, validation, and test data.
        In other words, this includes unknown/unseen items.

    uid_map: int
        Global mapping of user ID-index.

    iid_map: int
        Global mapping of item ID-index.
    """

    def __init__(self, name, trainable=True, verbose=False):
        super().__init__(name=name, trainable=trainable, verbose=verbose)

    def score(self, user_idx, history_items, **kwargs):
        """Predict the scores for all items based on input history items

        Parameters
        ----------
        history_items: list of lists
            The list of history items in sequential manner for next-item prediction.

        Returns
        -------
        res : a Numpy array
            Relative scores of all known items

        """
        raise NotImplementedError("The algorithm is not able to make score prediction!")
