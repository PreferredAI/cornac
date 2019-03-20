# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from . import FeatureModule


class ImageModule(FeatureModule):
    """Image module

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._id_image = kwargs.get('id_image', None)
        self._id_path = kwargs.get('id_path', None)
        self.data_image = None
        self.data_path = None

    @property
    def data_image(self):
        return self.__data_image

    @data_image.setter
    def data_image(self, input_image):
        self.__data_image = input_image

    def build(self, global_id_map):
        """Build the model based on provided list of ordered ids
        """
        FeatureModule.build(self, global_id_map)

    def batch_image(self, batch_ids,
                    target_size=(256, 256),
                    color_mode='rgb',
                    interpolation='nearest'):
        """Return batch of images corresponding to provided batch_ids
        """
        pass
