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

import os
from typing import List
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
from operator import itemgetter
import numpy as np

from . import FeatureModality


class TransformersTextModality(FeatureModality):
    """
    Transformer text modality wrapped around SentenceTrasformer library.
    https://huggingface.co/sentence-transformers.

    Parameters
    ----------
    corpus: List[str], default = None
        List of user/item texts that the indices are aligned with `ids`.
    """

    def __init__(
            self,
            corpus: List[str] = None,
            ids: List = None,
            preencode: bool = False,
            model_name_or_path: str = 'paraphrase-MiniLM-L6-v2',
            **kwargs):

        super().__init__(ids=ids, **kwargs)
        self.corpus = corpus
        self.model = SentenceTransformer(model_name_or_path)
        self.output_dim = self.model[-1].pooling_output_dimension
        self.preencode = preencode
        self.preencoded = False

    def preencode_entire_corpus(self):
        """
        Pre-encode the entire corpus. This is useful so that we don't have to do
        it on the fly in training. Might take significant time to pre-encode
        larger datasets.
        """
        
        path = 'temp/encoded_corpus.npy'
        id_path = "temp/encoded_corpus_ids.npy"

        if os.path.exists(path) and os.path.exists(id_path):
            saved_ids = np.load(id_path)
            if saved_ids == self.ids:
                self.features = np.load(path)
                self.preencoded = True
            else:
                assert self.preencoded is False
                print("The ids of the saved encoded corpus do not match the current ids. Re-encoding the corpus.")
        
        if not self.preencoded:
            print("Pre-encoding the entire corpus. This might take a while.")
            self.features = self.model.encode(self.corpus, convert_to_tensor=True)
            self.preencoded = True
            os.makedirs("temp", exist_ok = True)
            np.save(path, self.features)
            np.save(id_path, self.ids)

    def build(self, id_map: OrderedDict, **kwargs):
        """
        Build the modality with the given global id_map.

        :param id_map: the global id map (train and test set)
        """
        if (self.ids is not None) and (id_map is not None):
            self._swap_text(id_map)
        return self

    def _swap_text(self, id_map: dict):
        """
        Swap the text in the corpus according to the id_map. That way we can
        access the corpus by index, where the index represents the item id.

        :param id_map: the global id map (train and test set and possibly
            validation set)
        """
        new_corpus = self.corpus.copy()
        new_ids = self.ids.copy()
        for old_idx, raw_id in enumerate(self.ids):
            new_idx = id_map.get(raw_id, None)
            if new_idx is None:
                continue
            assert new_idx < len(self.corpus)
            new_corpus[new_idx] = self.corpus[old_idx]
            new_ids[new_idx] = raw_id
        self.corpus = new_corpus
        self.ids = new_ids

        if self.preencode:
            self.preencode_entire_corpus()


    def batch_encode(self, ids: List[int]):
        """
        Batch encode on the fly the list of item ids

        Parameters
        ----------
        ids: List[int]
            List of item ids to encode.
        """

        text_batch = list(itemgetter(*ids)(self.corpus))
        encoded = self.model.encode(text_batch, convert_to_tensor=True)

        return encoded