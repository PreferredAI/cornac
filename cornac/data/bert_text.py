import os
from typing import List, Dict, Callable, Union
from collections import defaultdict, Counter, OrderedDict
import string
import pickle
import re
import scipy
from sentence_transformers import SentenceTransformer, util
from operator import itemgetter

import numpy as np
import scipy.sparse as sp
import torch

from . import FeatureModality
from .modality import fallback_feature
from ..utils import normalize


class BertTextModality(FeatureModality):
    """Text modality using bert sentence encoder

    Parameters
    ----------
    corpus: List[str], default = None
        List of user/item texts that the indices are aligned with `ids`.
    """

    def __init__(self, corpus: List[str] = None, ids: List = None, preencode: bool = False, **kwargs):
        super().__init__(ids=ids, **kwargs)
        self.corpus = corpus
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.output_dim = self.model[-1].pooling_output_dimension
        self.preencode = preencode

    def preencode_entire_corpus(self):
        """
        Pre-encode the entire corpus. This is useful so that we don't have to do
        it on the fly in training. Might take significant time to pre-encode
        larger datasets.
        """
        
        path = 'temp/encoded_corpus.pt'
        try:
            self.encoded_corpus = torch.load(path)
            self.preencoded = True
        except FileNotFoundError:
            self.encoded_corpus = self.model.encode(self.corpus, convert_to_tensor=True)
            self.preencoded = True
            os.makedirs("temp", exist_ok = True) 
            torch.save(self.encoded_corpus, path)

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
        """

        text_batch = list(itemgetter(*ids)(self.corpus))
        encoded = self.model.encode(text_batch, convert_to_tensor=True)

        return encoded