# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np


def fallback_feature(func):
    """Decorator to fallback to `batch_feature` in FeatureModule
    """
    def wrapper(self, *args, **kwargs):
        if self.features is not None:
            ids = args[0] if len(args) > 0 else kwargs['batch_ids']
            return FeatureModule.batch_feature(self, batch_ids=ids)
        else:
            return func(self, *args, **kwargs)

    return wrapper


class Module:
    """Module
    """

    def __init__(self, **kwargs):
        pass


class FeatureModule(Module):
    """FeatureModule

    Parameters
    ----------
    features: numpy.ndarray or scipy.sparse.csr_matrix, default = None
        Numpy 2d-array that the row indices are aligned with user/item in `ids`.

    ids: List, default = None
        List of user/item ids that the indices are aligned with `corpus`.
        If None, the indices of provided `features` will be used as `ids`.
    """

    def __init__(self, features=None, ids=None, normalized=False, **kwargs):
        super().__init__(**kwargs)
        self.features = features
        self.ids = ids
        self._normalized = normalized

    @property
    def features(self):
        """Return the whole feature matrix
        """
        return self.__features

    @features.setter
    def features(self, input_features):
        if input_features is not None:
            assert len(input_features.shape) == 2
        self.__features = input_features

    @property
    def feature_dim(self):
        """Return the dimensionality of the feature vectors
        """
        return self.features.shape[1]

    def _swap_feature(self, id_map):
        if self.ids is None:
            self.ids = np.arange(self.features.shape[0])

        for old_idx, raw_id in enumerate(self.ids):
            new_idx = id_map.get(raw_id, None)
            if new_idx is None:
                continue
            assert new_idx < self.features.shape[0]
            self.features[[new_idx, old_idx]] = self.features[[old_idx, new_idx]]

    def build(self, id_map=None):
        """Build the feature matrix.
        Features will be swapped if the id_map is provided
        """
        if self.features is None:
            return

        if id_map is not None:
            self._swap_feature(id_map)

        if self._normalized:
            self.features = self.features - np.min(self.features)
            self.features = self.features / (np.max(self.features) + 1e-10)

    def batch_feature(self, batch_ids):
        """Return a matrix (batch of feature vectors) corresponding to provided batch_ids
        """
        assert self.features is not None
        return self.features[batch_ids]
