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

import numpy as np


class Modality:
    """Generic class of Modality to extend from
    """

    def __init__(self, **kwargs):
        pass

def fallback_feature(func):
    """Decorator to fallback to `batch_feature` in FeatureModality
    """

    def wrapper(self, *args, **kwargs):
        if self.features is not None:
            ids = args[0] if len(args) > 0 else kwargs['batch_ids']
            return FeatureModality.batch_feature(self, batch_ids=ids)
        else:
            return func(self, *args, **kwargs)

    return wrapper


class FeatureModality(Modality):
    """Modality that contains features in general

    Parameters
    ----------
    features: numpy.ndarray or scipy.sparse.csr_matrix, default = None
        Numpy 2d-array that the row indices are aligned with user/item in `ids`.

    ids: List, default = None
        List of user/item ids that the indices are aligned with `corpus`.
        If None, the indices of provided `features` will be used as `ids`.

    normalized: bool, default = False
        Whether the features will be normalized using min-max normalization.
    """

    def __init__(self, features=None, ids=None, normalized=False, **kwargs):
        super().__init__(**kwargs)
        self.features = features
        self.ids = ids
        self.normalized = normalized

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
        new_feats = np.copy(self.features)
        new_ids = self.ids.copy()
        for old_idx, raw_id in enumerate(self.ids):
            new_idx = id_map.get(raw_id, None)
            if new_idx is None:
                continue
            assert new_idx < new_feats.shape[0]
            new_feats[new_idx] = self.features[old_idx]
            new_ids[new_idx] = raw_id
        self.features = new_feats
        self.ids = new_ids

    def build(self, id_map=None, **kwargs):
        """Build the feature matrix.
        Features will be swapped if the id_map is provided
        """
        if self.features is None:
            return

        if (self.ids is not None) and (id_map is not None):
            self._swap_feature(id_map)

        if self.normalized:
            self.features = self.features - np.min(self.features)
            self.features = self.features / (np.max(self.features) + 1e-10)

        return self

    def batch_feature(self, batch_ids):
        """Return a matrix (batch of feature vectors) corresponding to provided batch_ids
        """
        assert self.features is not None
        return self.features[batch_ids]
