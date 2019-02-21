# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import numpy as np


class ImageModule:
    """Image module

    """

    def __init__(self, **kwargs):
        self._id_feature = kwargs.get('id_feature', None)
        self._id_image = kwargs.get('id_image', None)
        self._id_path = kwargs.get('id_path', None)

        self.data_feature = None
        self.data_image = None
        self.data_path = None
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

    @property
    def data_image(self):
        return self.__data_image

    @data_image.setter
    def data_image(self, input_image):
        self.__data_image = input_image

    def _build_feature(self, ordered_ids):
        """Build data_feature matrix based on provided list of ordered ids
        """
        if len(self._id_feature) == 0:
            return
        self.data_feature = np.zeros((len(ordered_ids), self.feature_dim))
        for idx, id in enumerate(ordered_ids):
            self.data_feature[idx] = self._id_feature[id]
            del self._id_feature[id]
        self._id_feature.clear()

    def build(self, ordered_ids):
        """Build the model based on provided list of ordered ids
        """
        self._build_feature(ordered_ids)

    def batch_feature(self, batch_ids):
        """Return a matrix (batch of feature vectors) corresponding to provided batch_ids
        """
        return self.data_feature[batch_ids]

    def batch_image(self, batch_ids,
                    target_size=(256, 256),
                    color_mode='rgb',
                    interpolation='nearest'):
        """Return batch of images corresponding to provided batch_ids
        """
        pass
