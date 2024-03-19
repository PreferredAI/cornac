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

from collections import OrderedDict
from typing import List
from cornac.data.modality import FeatureModality
import os
from PIL.JpegImagePlugin import JpegImageFile
import torch
from torchvision import transforms, models
from torchvision.models._api import WeightsEnum


class TransformersVisionModality(FeatureModality):
    """
    Transformer vision modality wrapped around the torchvision ViT Transformer.

    Parameters
    ----------
    corpus: List[JpegImageFile], default = None
        List of user/item texts that the indices are aligned with `ids`.
    """

    def __init__(
        self,
        images: List[JpegImageFile] = None,
        ids: List = None,
        preencode: bool = False,
        model_weights: WeightsEnum = models.ViT_H_14_Weights.DEFAULT,
        **kwargs
    ):

        super().__init__(ids=ids, **kwargs)
        self.images = images
        self.model = models.vit_h_14(weights=model_weights)
        # suppress the classification piece
        self.model.heads = torch.nn.Identity()
        self.model.eval()

        self.image_size = (self.model.image_size, self.model.image_size)
        self.image_to_tensor_transformer = transforms.Compose(
            [
                transforms.ToTensor()
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize pixel values
            ]
        )

        self.preencode = preencode
        self.preencoded = False
        self.batch_size = 50
        
        if self.preencode:
            self.preencode_images()

    def preencode_images(self):
        """
        Pre-encode the entire image library. This is useful so that we don't
        have to do it on the fly in training. Might take significant time to
        pre-encode.
        """

        path = "temp/encoded_images.pt"
        id_path = "temp/encoded_images_ids.pt"

        if os.path.exists(path) and os.path.exists(id_path):
            saved_ids = torch.load(id_path)
            if saved_ids == self.ids:
                self.features = torch.load(path)
                self.preencoded = True
            else:
                assert self.preencoded is False
                print(
                    "The ids of the saved encoded images do not match the current ids. Re-encoding the images."
                )

        if not self.preencoded:
            print("Pre-encoding the entire image library. This might take a while.")
            self._encode_images()
            self.preencoded = True
            os.makedirs("temp", exist_ok=True)
            torch.save(self.features, path)
            torch.save(self.ids, id_path)

    def _encode_images(self):
        """
        Encode all images in the library.
        """
        for i in range(len(self.images) // self.batch_size + 1):
            tensor_batch = self.transform_images_to_torch_tensor(
                self.images[i * self.batch_size : (i + 1) * self.batch_size]
            )
            with torch.no_grad():
                encoded_batch = self.model(tensor_batch)

            if i == 0:
                self.features = encoded_batch
            else:
                self.features = torch.cat((self.features, encoded_batch), 0)

    def transform_images_to_torch_tensor(
        self, images: List[JpegImageFile]
    ) -> torch.Tensor:
        """
        Transorms a list of PIL images to a torch tensor batch.

        Parameters
        ----------
        images: List[PIL.Image]
            List of PIL images to be transformed to torch tensor.
        """
        for i, img in enumerate(images):
            if img.size != self.image_size:
                img = img.resize(self.image_size)

            tensor = self.image_to_tensor_transformer(img)
            tensor = tensor.unsqueeze(0)
            if i == 0:
                tensor_batch = tensor
            else:
                tensor_batch = torch.cat((tensor_batch, tensor), 0)

        return tensor_batch

    def batch_encode(self, ids: List[int]):
        """
        Batch encode on the fly the photos for the list of item ids.

        Parameters
        ----------
        ids: List[int]
            List of item ids to encode.
        """
        tensor_batch = self.transform_images_to_torch_tensor(self.images[ids])
        with torch.no_grad():
            encoded_batch = self.model(tensor_batch)

        return encoded_batch
