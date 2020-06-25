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

from . import FeatureModality


class ImageModality(FeatureModality):
    """Image modality

    Parameters
    ----------
    images: Union[List, numpy.ndarray], optional
        A list or tensor of images that the row indices are
        aligned with user/item in `ids`.

    paths: List[str], optional
        A list of paths, to images stored on disk, which
        the row indices are aligned with user/item in `ids`..

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.images = kwargs.get('images', None)
        self.paths = kwargs.get('paths', None)

    def build(self, id_map=None, **kwargs):
        """Build the model based on provided list of ordered ids

        Parameters
        ----------
        id_map: dict, optional
            A dictionary holds mapping from original ids to
            mapped integer indices of users/items.

        Returns
        -------
        image_modality: :obj:`<cornac.data.ImageModality>`
            An object of type `ImageModality`.

        """
        super().build(id_map=id_map)
        return self

    def batch_image(self, batch_ids,
                    target_size=(256, 256),
                    color_mode='rgb',
                    interpolation='nearest'):
        """Return batch of images corresponding to provided batch_ids

        Parameters
        ----------
        batch_ids: Union[List, numpy.array], required
            An array containing the ids of rows of images to be returned.

        target_size: tuple, optional, default: (256, 256)
            Size (width, height) of returned images to be resized.

        color_mode: str, optional, default: 'rgb'
            Color mode of returned images.

        interpolation: str, optional, default: 'nearest'
            Method used for interpolation when resize images.
            Options are OpenCV supported methods.

        Returns
        -------
        res: numpy.ndarray
            Batch of images corresponding to input `batch_ids`.
        """
        raise NotImplementedError
