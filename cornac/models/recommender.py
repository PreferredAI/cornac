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

import os
import copy
import inspect
import pickle
from glob import glob
from datetime import datetime

import numpy as np

from ..exception import ScoreException
from ..utils.common import intersects, clip


class Recommender:
    """Generic class for a recommender model. All recommendation models should inherit from this class 
    
    Parameters
    ----------------
    name: str, required
        The name of the recommender model

    trainable: boolean, optional, default: True
        When False, the model is not trainable

    """

    def __init__(self, name, trainable=True, verbose=False):
        self.name = name
        self.trainable = trainable
        self.verbose = verbose
        self.train_set = None
        self.val_set = None
        # attributes to be ignored when being saved
        self.ignored_attrs = ["train_set", "val_set"]

    def reset_info(self):
        self.best_value = -np.Inf
        self.best_epoch = 0
        self.current_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k in self.ignored_attrs:
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

    def save(self, save_dir=None):
        """Save a recommender model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

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
        self.reset_info()
        self.train_set = train_set.reset()
        self.val_set = None if val_set is None else val_set.reset()
        return self

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
        """Overwrite this function if your algorithm has special treatment for cold-start problem

        """
        return self.train_set.global_mean

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
            rating_pred = clip(
                values=rating_pred,
                lower_bound=self.train_set.min_rating,
                upper_bound=self.train_set.max_rating,
            )

        return rating_pred

    def rank(self, user_idx, item_indices=None):
        """Rank all test items for a given user.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform item raking.

        item_indices: 1d array, optional, default: None
            A list of candidate item indices to be ranked by the user.
            If `None`, list of ranked known item indices and their scores will be returned.

        Returns
        -------
        (item_rank, item_scores): tuple
            `item_rank` contains item indices being ranked by their scores.
            `item_scores` contains scores of items corresponding to their indices in the `item_indices` input.
        """
        # obtain item scores from the model
        try:
            known_item_scores = self.score(user_idx)
        except ScoreException:
            known_item_scores = (
                np.ones(self.train_set.total_items) * self.default_score()
            )

        # check if the returned scores also cover unknown items
        # if not, all unknown items will be given the MIN score
        if len(known_item_scores) == self.train_set.total_items:
            all_item_scores = known_item_scores
        else:
            all_item_scores = np.ones(self.train_set.total_items) * np.min(
                known_item_scores
            )
            all_item_scores[: self.train_set.num_items] = known_item_scores

        # rank items based on their scores
        if item_indices is None:
            item_scores = all_item_scores[: self.train_set.num_items]
            item_rank = item_scores.argsort()[::-1]
        else:
            item_scores = all_item_scores[item_indices]
            item_rank = np.array(item_indices)[item_scores.argsort()[::-1]]

        return item_rank, item_scores

    def monitor_value(self):
        """Calculating monitored value used for early stopping on validation set (`val_set`).
        This function will be called by `early_stop()` function.
        Note: `val_set` could be `None` thus it needs to be checked before usage.

        Returns
        -------
        :raise NotImplementedError
        """
        raise NotImplementedError()

    def early_stop(self, min_delta=0.0, patience=0):
        """Check if training should be stopped when validation loss has stopped improving.

        Parameters
        ----------
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
        current_value = self.monitor_value()
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
