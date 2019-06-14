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
        self.images = kwargs.get('images', None)
        self.paths = kwargs.get('paths', None)

    def build(self, id_map=None):
        """Build the model based on provided list of ordered ids
        """
        super().build(id_map)
        return self

    def batch_image(self, batch_ids,
                    target_size=(256, 256),
                    color_mode='rgb',
                    interpolation='nearest'):
        """Return batch of images corresponding to provided batch_ids
        """
        raise NotImplementedError