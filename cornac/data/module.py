# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np


class Module:
    """Module
    """

    def __init__(self, **kwargs):
        pass


class FeatureModule(Module):
    """FeatureModule
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._id_feature = kwargs.get('id_feature', None)
        self._normalized = kwargs.get('normalized', False)

        self.features = None
        if self._id_feature is not None:
            self.__feat_dim = next(iter(self._id_feature.values())).shape[0]

    @property
    def features(self):
        """Return the whole feature matrix
        """
        return self.__features

    @features.setter
    def features(self, input_features):
        self.__features = input_features

    @property
    def feat_dim(self):
        """Return the dimensionality of the feature vectors
        """
        return self.__feat_dim

    def build(self, global_id_map):
        """Build the features based on provided global id map
        """
        if self._id_feature is None:
            return

        self.features = np.zeros((len(global_id_map), self.feat_dim))
        for mapped_id, raw_id in enumerate(global_id_map.keys()):
            self.features[mapped_id] = self._id_feature[raw_id]
        if self._normalized:
            self.features = self.features - np.min(self.features)
            self.features = self.features / (np.max(self.features) + 1e-10)

        self._id_feature.clear()

    def batch_feat(self, batch_ids):
        """Return a matrix (batch of feature vectors) corresponding to provided batch_ids
        """
        return self.features[batch_ids]
