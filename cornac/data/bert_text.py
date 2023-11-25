from typing import List, Dict, Callable, Union
from collections import defaultdict, Counter, OrderedDict
import string
import pickle
import re
from sentence_transformers import SentenceTransformer, util
from operator import itemgetter

import numpy as np
import scipy.sparse as sp

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

    def __init__(self,
                corpus: List[str] = None,
                ids: List = None,
                **kwargs):

                super().__init__(ids=ids, **kwargs)
                self.corpus = corpus
                self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    def build(self):
        """
        """
        return self
    
    def batch_encode(self, ids: List[int]):
        """
        """
        text_batch = list(itemgetter(*ids)(self.corpus))
        encoded = self.model.encode(text_batch, convert_to_tensor=True)

        return encoded