# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

from . import FeatureModule
from typing import List, Dict, Callable
from collections import defaultdict, Counter
import pickle
import numpy as np
import itertools
import re
import string

PAD, UNK, BOS, EOS = '<PAD>', '<UNK>', '<BOS>', '<EOS>'
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


class Tokenizer():
    """
    Generic class for other subclasses to extend from. This typically
    either splits text into word tokens or character tokens.
    """

    def tokenize(self, t: str) -> List[str]:
        """
        Splitting text into tokens.

        Returns
        -------
        tokens : ``List[str]``
        """
        raise NotImplementedError

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Splitting a corpus with multiple text documents.

        Returns
        -------
        tokens : ``List[List[str]]``
        """
        raise NotImplementedError


def rm_tags(t: str) -> str:
    """
    Remove html tags.
    e,g, rm_tags("<i>Hello</i> <b>World</b>!") -> "Hello World".
    """
    return re.sub('<([^>]+)>', '', t)


def rm_numeric(t: str) -> str:
    """
    Remove digits from `t`.
    """
    return re.sub('[0-9]+', ' ', t)


def rm_punctuation(t: str) -> str:
    """
    Remove "!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~" from t.
    """
    return t.translate(str.maketrans('', '', string.punctuation))


def rm_dup_spaces(t: str) -> str:
    """
    Remove duplicate spaces in `t`.
    """
    return re.sub('(\s)+', ' ', t)


DEFAULT_PRE_RULES = [lambda t: t.lower(), rm_tags, rm_numeric, rm_punctuation, rm_dup_spaces]


class BaseTokenizer(Tokenizer):
    """
    A base tokenizer use a provided delimiter `sep` to split text.
    """

    def __init__(self, sep=' ', pre_rules: List[Callable[[str], str]] = None):
        self.sep = sep
        self.pre_rules = DEFAULT_PRE_RULES if pre_rules is None else pre_rules

    def tokenize(self, t: str) -> List[str]:
        """
        Splitting text into tokens.

        Returns
        -------
        tokens : ``List[str]``
        """
        for rule in self.pre_rules:
            t = rule(t)
        tokens = t.split(self.sep)
        return tokens

    # TODO: this function can be parallelized
    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Splitting a corpus with multiple text documents.

        Returns
        -------
        tokens : ``List[List[str]]``
        """
        return [self.tokenize(t) for t in texts]


class Vocabulary():
    """
    Vocabulary basically contains mapping between numbers and tokens and vice versa.
    """

    def __init__(self, idx2tok: List[str]):
        self.idx2tok = self._add_special_tokens(idx2tok)
        self.tok2idx = defaultdict(int, {tok: idx for idx, tok in enumerate(self.idx2tok)})

    @staticmethod
    def _add_special_tokens(idx2tok: List[str]) -> List[str]:
        for tok in reversed(SPECIAL_TOKENS):  # <PAD>:0, '<UNK>':1, '<BOS>':2, '<EOS>':3
            if tok in idx2tok:
                idx2tok.remove(tok)
            idx2tok.insert(0, tok)
        return idx2tok

    @property
    def size(self):
        return len(self.idx2tok)

    def to_idx(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of `tokens` to their integer indices.
        """
        return [self.tok2idx.get(tok, 1) for tok in tokens]  # 1 is <UNK> idx

    def to_text(self, indices: List[int], sep=' ') -> List[str]:
        """
        Convert a list of integer `indices` to their tokens.
        """
        return sep.join([self.idx2tok[i] for i in indices]) if sep is not None else [self.idx2tok[i] for i in indices]

    def save(self, path):
        """
        Save idx2tok into a pickle file.
        """
        pickle.dump(self.idx2tok, open(path, 'wb'))

    @classmethod
    def from_tokens(cls, tokens: List[str], max_vocab: int = None, min_freq: int = 1) -> 'Vocabulary':
        """
        Build a vocabulary from list of tokens.
        """
        freq = Counter(tokens)
        idx2tok = [tok for tok, cnt in freq.most_common(max_vocab) if cnt >= min_freq]
        return cls(idx2tok)

    @classmethod
    def load(cls, path):
        """
        Load a vocabulary from `path` to a pickle file.
        """
        return cls(pickle.load(open(path, 'rb')))


class TextModule(FeatureModule):
    """Text module

    Parameters
    ----------
    id_text: Dict, optional, default = None
        A dictionary contains mapping between user/item id to their text.

    tokenizer: Tokenizer, optional, default = None
        Tokenizer for text splitting. If None, the BaseTokenizer will be used.

    vocab: Vocabulary, optional, default = None
        Vocabulary of tokens. It contains mapping between tokens to their
        integer ids and vice versa.

    max_vocab: int, optional, default = None
        The maximum size of the vocabulary.
        If vocab is provided, this will be ignored.

    min_freq: int, default = 1
        The minimum frequency of tokens to be included into vocabulary.
        If vocab is provided, this will be ignored.

    """

    def __init__(self,
                 id_text: Dict = None,
                 tokenizer: Tokenizer = None,
                 vocab: Vocabulary = None,
                 max_vocab: int = None,
                 min_freq: int = 1,
                 **kwargs):
        super().__init__(**kwargs)

        self._id_text = id_text
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.sequences = None

    def _build_text(self, global_id_map: Dict):
        """Build the text based on provided global id map
        """
        if self._id_text is None:
            return

        if self.tokenizer is None:
            self.tokenizer = BaseTokenizer()

        # Tokenize texts
        self.sequences = []
        mapped2raw = {mapped_id: raw_id for raw_id, mapped_id in global_id_map.items()}
        for mapped_id in range(len(global_id_map)):
            raw_id = mapped2raw[mapped_id]
            text = self._id_text[raw_id]
            self.sequences.append(self.tokenizer.tokenize(text))
            del self._id_text[raw_id]
        del self._id_text

        if self.vocab is None:
            self.vocab = Vocabulary.from_tokens(tokens=list(itertools.chain(*self.sequences)),
                                                max_vocab=self.max_vocab,
                                                min_freq=self.min_freq)

        # Map tokens into integer ids
        for i, seq in enumerate(self.sequences):
            self.sequences[i] = self.vocab.to_idx(seq)

    def build(self, global_id_map):
        """Build the model based on provided list of ordered ids
        """
        FeatureModule.build(self, global_id_map)
        self._build_text(global_id_map)

    def batch_seq(self, batch_ids, max_length=None):
        """Return a numpy matrix of text sequences containing token ids with size=(len(batch_ids), max_length).
        If max_length=None, it will be inferred based on retrieved sequences.
        """
        if self.sequences is None:
            raise ValueError('self.sequences is required but None!')

        if max_length is None:
            max_length = max(len(self.sequences[mapped_id]) for mapped_id in batch_ids)

        seq_mat = np.zeros((len(batch_ids), max_length), dtype=np.int)
        for i, mapped_id in enumerate(batch_ids):
            idx_seq = self.sequences[mapped_id][:max_length]
            for j, idx in enumerate(idx_seq):
                seq_mat[i, j] = idx

        return seq_mat

    def batch_bow(self, batch_ids):
        """Return matrix of bag-of-words corresponding to provided batch_ids
        """
        raise NotImplementedError

    def batch_freq(self, batch_ids):
        """Return matrix of word frequencies corresponding to provided batch_ids
        """
        raise NotImplementedError

    def batch_tfidf(self, batch_ids):
        """Return matrix of TF-IDF features corresponding to provided batch_ids
        """
        raise NotImplementedError
