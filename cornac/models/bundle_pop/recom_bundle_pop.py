# Copyright 2026 The Cornac Authors. All Rights Reserved.
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

import numpy as np
import scipy.sparse as sp

from ..recommender import Recommender
from ...exception import ScoreException


def _binary(matrix):
    return None if matrix is None else (matrix != 0).astype(np.float64).tocsr()


class BundlePop(Recommender):
    """Bundle Popularity. Non-learned baselines for bundle recommendation,
    completion and construction, computed from training statistics only.

    ``fit`` accepts a :class:`cornac.data.BundleDataset` (recommendation; targets
    are catalog bundles) or a :class:`cornac.data.BundleConstructionDataset`
    (completion and construction; targets are the constituent items of a bundle
    instance). On construction data, full training bundles are the training
    instances not in ``partial_bundles``.

    Parameters
    ----------
    name: string, default: 'BundlePop'
        The name of the recommender model.

    popularity: string, default: 'item'
        Scoring rule. On a ``BundleDataset``:

        - ``'bundle'``: number of training users of the bundle (as MostPop).
        - ``'item'``: mean popularity of the bundle's items, i.e. their
          ``user_item_matrix`` degree if available, else their
          ``bundle_item_matrix`` degree. Bundles without feedback are scored.
        - ``'user'``: number of the bundle's items in the user's history
          (user-item feedback and items of the user's training bundles) divided
          by the bundle size. Ties are broken by bundle popularity scaled below
          1 / max_size^2, the smallest gap between two distinct fractions.

        On a ``BundleConstructionDataset``:

        - ``'bundle'``: sum, over the full training bundles containing the item,
          of the bundle's number of users in ``user_bundle_matrix`` (1 per
          bundle when ``user_bundle_matrix`` is None).
        - ``'item'``: number of full training bundles containing the item.
        - ``'user'``: occurrences of the item in the owner's history (one per
          user-item observation and per training bundle of the owner). Ties are
          broken by item popularity. Instances without an owner use ``'item'``.
        - ``'cooccurrence'``: sum, over the given items, of the number of full
          training bundles containing both the given item and the item. Ties
          are broken by item popularity. Instances without given items use
          ``'item'``.

        Tie-breakers never outrank a difference in the primary score.
        ``'cooccurrence'`` requires a ``BundleConstructionDataset``.
    """

    def __init__(self, name="BundlePop", popularity="item"):
        super().__init__(name=name)
        if popularity not in ("bundle", "item", "user", "cooccurrence"):
            raise ValueError("popularity must be bundle, item, user or cooccurrence")
        self.popularity = popularity

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.BundleDataset` or :obj:`cornac.data.BundleConstructionDataset`, required
            Training bundle data.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            Not used.

        Returns
        -------
        self : object
        """
        from ...data import BundleConstructionDataset, BundleDataset

        self.pop_scores = None
        if isinstance(train_set, BundleDataset):
            if self.popularity == "cooccurrence":
                raise ValueError(
                    "cooccurrence popularity requires a BundleConstructionDataset"
                )
            Recommender.fit(self, train_set, val_set)
            self.owner_indices = None
            self.user_bundle = _binary(train_set.csr_matrix)
            self.bundle_item = _binary(train_set.bundle_item_matrix)
            self.user_item = _binary(train_set.user_item_matrix)
            self.bundle_sizes = np.ediff1d(self.bundle_item.indptr)
            bundle_pop = np.ediff1d(train_set.csc_matrix.indptr)
            if self.popularity == "bundle":
                self.pop_scores = bundle_pop
            elif self.popularity == "item":
                degree = (
                    self.bundle_item if self.user_item is None else self.user_item
                ).sum(axis=0)
                self.pop_scores = (
                    self.bundle_item @ np.asarray(degree).ravel() / self.bundle_sizes
                )
            self.tie_breaker = (
                bundle_pop / (bundle_pop.max() + 1) / self.bundle_sizes.max() ** 2
            )
        elif isinstance(train_set, BundleConstructionDataset):
            Recommender.fit(self, train_set, val_set)
            self.owner_indices = train_set.owner_indices
            self.user_bundle = _binary(train_set.user_bundle_matrix)
            self.user_item = _binary(train_set.user_item_matrix)
            self.instance_items = _binary(train_set.csr_matrix)
            full = np.ones(train_set.num_users)
            full[np.asarray(train_set.partial_bundles, dtype=int)] = 0
            self.bundle_item = (sp.diags(full) @ self.instance_items).tocsr()
            self.bundle_item.eliminate_zeros()
            self.item_pop = np.asarray(self.bundle_item.sum(axis=0)).ravel()
            if self.popularity == "bundle":
                weights = (
                    np.ones(train_set.num_users)
                    if self.user_bundle is None
                    else np.asarray(self.user_bundle.sum(axis=0)).ravel()
                )
                self.pop_scores = self.bundle_item.T @ weights
            elif self.popularity == "item":
                self.pop_scores = self.item_pop
            self.tie_breaker = self.item_pop / (self.item_pop.max() + 1)
        else:
            raise ValueError(
                "train_set must be a BundleDataset or BundleConstructionDataset"
            )
        return self

    def _history(self, row):
        """Item counts of a user's (or owner's) training bundles and item feedback."""
        history = np.zeros(self.bundle_item.shape[1])
        if self.user_bundle is not None:
            history += (self.user_bundle[row] @ self.bundle_item).toarray().ravel()
        if self.user_item is not None:
            history += self.user_item[row].toarray().ravel()
        return history

    def _personal_scores(self, user_idx):
        if self.owner_indices is None:  # BundleDataset: overlap fraction per bundle
            overlap = self.bundle_item @ (self._history(user_idx) > 0).astype(
                np.float64
            )
            return overlap / self.bundle_sizes + self.tie_breaker
        if self.popularity == "cooccurrence":
            given = self.instance_items[user_idx].toarray().ravel()
            if not given.any():
                return self.item_pop
            # B^T (B x) sums co-counts over given items; subtracting x * pop zeroes the diagonal.
            counts = (
                self.bundle_item.T @ (self.bundle_item @ given) - given * self.item_pop
            )
            return counts + self.tie_breaker
        owner = self.owner_indices[user_idx]
        if owner < 0:
            return self.item_pop
        return self._history(owner) + self.tie_breaker

    def score(self, user_idx, item_idx=None):
        """Predict the scores of a user (or bundle instance) for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user, or of the bundle instance on construction data.

        item_idx: int, optional, default: None
            The index of the target (bundle or item) to score.
            If None, scores for all known targets will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores of the target or of all known targets.
        """
        if self.pop_scores is not None:
            scores = self.pop_scores
        elif self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)
        else:
            scores = self._personal_scores(user_idx)
        if item_idx is None:
            return scores
        if not self.knows_item(item_idx):
            raise ScoreException(
                "Can't make score prediction for (user_id=%d, item_id=%d)"
                % (user_idx, item_idx)
            )
        return scores[item_idx]
