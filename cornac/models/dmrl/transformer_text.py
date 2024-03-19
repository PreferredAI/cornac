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

import torch

from cornac.data.modality import FeatureModality


class TransformersTextModality(FeatureModality):
    """
    Transformer text modality wrapped around SentenceTransformer library.
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
        
        if self.preencode:
            self.preencode_entire_corpus()

    def preencode_entire_corpus(self):
        """
        Pre-encode the entire corpus. This is useful so that we don't have to do
        it on the fly in training. Might take significant time to pre-encode
        larger datasets.
        """
        
        path = "temp/encoded_corpus.pt"
        id_path = "temp/encoded_corpus_ids.pt"

        if os.path.exists(path) and os.path.exists(id_path):
            saved_ids = torch.load(id_path)
            try:
                if saved_ids == self.ids:
                    self.features = torch.load(path)
                    self.preencoded = True
                else:
                    assert self.preencoded is False
            except:  # noqa: E722
                print("The ids of the saved encoded corpus do not match the current ids. Re-encoding the corpus.")
        
        if not self.preencoded:
            print("Pre-encoding the entire corpus. This might take a while.")
            self.features = self.model.encode(self.corpus, convert_to_tensor=True)
            self.preencoded = True
            os.makedirs("temp", exist_ok = True)
            torch.save(self.features, path)
            torch.save(self.ids, id_path)

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
