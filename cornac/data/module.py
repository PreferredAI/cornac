# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np

class Module:
    """Module

    """

    def __init__(self, **kwargs):
        self._id_feature = kwargs.get('id_feature', None)
        self._normalized = kwargs.get('normalized', False)

        self.data_feature = None
        if self._id_feature is not None:
            first_id = list(self._id_feature.keys())[0]
            self.feature_dim = self._id_feature[first_id].shape[0]

    @property
    def data_feature(self):
        return self.__data_feature

    @data_feature.setter
    def data_feature(self, input_feature):
        self.__data_feature = input_feature

    @property
    def feature_dim(self):
        return self.__feature_dim

    @feature_dim.setter
    def feature_dim(self, input_dim):
        self.__feature_dim = input_dim

    def _build_feature(self, ordered_ids):
        """Build data_feature matrix based on provided list of ordered ids
        """
        if self._id_feature is None:
            return

        self.data_feature = np.zeros((len(ordered_ids), self.feature_dim))
        for map_id, raw_id in enumerate(ordered_ids.keys()):
            self.data_feature[map_id] = self._id_feature[raw_id]
        if self._normalized:
            self.data_feature = self.data_feature - np.min(self.data_feature)
            self.data_feature = self.data_feature / (np.max(self.data_feature) + 1e-10)

        self._id_feature.clear()

    def build(self, ordered_ids):
        """Build the model based on provided list of ordered ids
        """
        self._build_feature(ordered_ids)

    def batch_feature(self, batch_ids):
        """Return a matrix (batch of feature vectors) corresponding to provided batch_ids
        """
        return self.data_feature[batch_ids]