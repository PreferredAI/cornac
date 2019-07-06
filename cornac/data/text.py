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

from typing import List, Dict, Callable, Union
from collections import defaultdict, Counter
import string
import pickle
import re

import numpy as np
import scipy.sparse as sp

from . import FeatureModule
from .module import fallback_feature

__all__ = ['Tokenizer',
           'BaseTokenizer',
           'Vocabulary',
           'CountVectorizer',
           'TextModule']

PAD, UNK, BOS, EOS = '<PAD>', '<UNK>', '<BOS>', '<EOS>'
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

ENGLISH_STOPWORDS = frozenset([
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone',
    'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst', 'amount',
    'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around',
    'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before',
    'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both',
    'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de',
    'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven',
    'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything',
    'everywhere', 'except', 'few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first', 'five', 'for',
    'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give',
    'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
    'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if',
    'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter',
    'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine',
    'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely',
    'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not',
    'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other',
    'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps',
    'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious',
    'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow',
    'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten',
    'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby',
    'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin', 'third', 'this', 'those', 'though',
    'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards',
    'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we',
    'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
    'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever',
    'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your',
    'yours', 'yourself', 'yourselves'])


def _validate_stopwords(stop_words):
    if stop_words == 'english':
        return ENGLISH_STOPWORDS
    elif isinstance(stop_words, str):
        raise ValueError("Invalid built-in stop-words list: %s" % stop_words)
    elif stop_words is None:
        return None
    else:
        return frozenset(stop_words)


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
    return re.sub(' {2,}', ' ', t)


DEFAULT_PRE_RULES = [lambda t: t.lower(), rm_tags, rm_numeric, rm_punctuation, rm_dup_spaces]


class BaseTokenizer(Tokenizer):
    """
    A base tokenizer use a provided delimiter `sep` to split text.
    """

    def __init__(self, sep: str = ' ',
                 pre_rules: List[Callable[[str], str]] = None,
                 stop_words: Union[List, str] = None):
        self.sep = sep
        self.pre_rules = DEFAULT_PRE_RULES if pre_rules is None else pre_rules
        self.stop_words = _validate_stopwords(stop_words)

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
        if self.stop_words is not None:
            tokens = [tok for tok in tokens if tok not in self.stop_words]
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

    def __init__(self, idx2tok: List[str], use_special_tokens: bool = False):
        self.use_special_tokens = use_special_tokens
        self.idx2tok = self._add_special_tokens(idx2tok) if use_special_tokens else idx2tok
        self.build_tok2idx()

    def build_tok2idx(self):
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
    def from_tokens(cls, tokens: List[str],
                    max_vocab: int = None,
                    min_freq: int = 1,
                    use_special_tokens: bool = False) -> 'Vocabulary':
        """
        Build a vocabulary from list of tokens.
        """
        freq = Counter(tokens)
        idx2tok = [tok for tok, cnt in freq.most_common(max_vocab) if cnt >= min_freq]
        return cls(idx2tok, use_special_tokens)

    @classmethod
    def from_sequences(cls, sequences: List[List[str]],
                       max_vocab: int = None,
                       min_freq: int = 1,
                       use_special_tokens: bool = False) -> 'Vocabulary':
        """
        Build a vocabulary from sequences (list of list of tokens).
        """
        return Vocabulary.from_tokens([tok for seq in sequences for tok in seq],
                                      max_vocab, min_freq, use_special_tokens)

    @classmethod
    def load(cls, path):
        """
        Load a vocabulary from `path` to a pickle file.
        """
        return cls(pickle.load(open(path, 'rb')))


class CountVectorizer():
    """Convert a collection of text documents to a matrix of token counts
    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    Parameters
    ----------
    tokenizer: Tokenizer, optional, default = None
        Tokenizer for text splitting. If None, the BaseTokenizer will be used.

    vocab: Vocabulary, optional, default = None
        Vocabulary of tokens. It contains mapping between tokens to their
        integer ids and vice versa.

    max_doc_freq: Union[float, int] = 1.0
        The maximum frequency of tokens appearing in documents to be excluded from vocabulary.
        If float, the value represents a proportion of documents, int for absolute counts.
        If `vocab` is not None, this will be ignored.

    min_freq: int, default = 1
        The minimum frequency of tokens to be included into vocabulary.
        If `vocab` is not None, this will be ignored.

    max_features : int, default=None
        If not None, build a vocabulary that only consider the top
        `max_features` ordered by term frequency across the corpus.
        If `vocab` is not None, this will be ignored.

    binary : boolean, default=False
        If True, all non zero counts are set to 1.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 vocab: Vocabulary = None,
                 max_doc_freq: Union[float, int] = 1.0,
                 min_freq: int = 1,
                 max_features: int = None,
                 binary: bool = False):
        self.tokenizer = BaseTokenizer() if tokenizer is None else tokenizer
        self.vocab = vocab
        self.max_doc_freq = max_doc_freq
        self.min_freq = min_freq
        if max_doc_freq < 0 or min_freq < 0:
            raise ValueError('negative value for max_doc_freq or min_freq')
        self.max_features = max_features
        if max_features is not None:
            if max_features <= 0:
                raise ValueError('max_features=%r, '
                                 'neither a positive integer nor None' % max_features)
        self.binary = binary

    def _limit_features(self, X: sp.csr_matrix, max_doc_count: int):
        """Remove too common features.
        Prune features that are non zero in more samples than max_doc_count
        and modifying the vocabulary.
        """
        if max_doc_count >= X.shape[0]:
            return X

        # Calculate a mask based on document frequencies
        doc_freq = np.bincount(X.indices, minlength=X.shape[1])
        term_indices = np.arange(X.shape[1])  # terms are already sorted based on frequency from Vocabulary
        mask = np.ones(len(doc_freq), dtype=bool)
        mask &= doc_freq <= max_doc_count

        if self.max_features is not None and mask.sum() > self.max_features:
            mask_indices = term_indices[mask][:self.max_features]
            new_mask = np.zeros(len(doc_freq), dtype=bool)
            new_mask[mask_indices] = True
            mask = new_mask

        for index in np.sort(np.where(np.logical_not(mask))[0])[::-1]:
            del self.vocab.idx2tok[index]
        self.vocab.build_tok2idx()  # rebuild the mapping

        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_freq or a higher max_doc_freq.")
        return X[:, kept_indices]

    def _count(self, sequences: List[List[str]]):
        """
        Create sparse feature matrix of document term counts
        Ignore SPECIAL_TOKENS if used from count matrix
        """
        data = []
        indices = []
        indptr = [0]
        for sequence in sequences:
            feature_counter = Counter()
            for token in sequence:
                if token not in self.vocab.tok2idx.keys():
                    continue
                idx = self.vocab.tok2idx[token]
                if self.vocab.use_special_tokens:
                    idx -= len(SPECIAL_TOKENS)
                feature_counter[idx] += 1

            indices.extend(feature_counter.keys())
            data.extend(feature_counter.values())
            indptr.append(len(indices))

        indices = np.asarray(indices, dtype=np.int)
        indptr = np.asarray(indptr, dtype=np.int)
        data = np.asarray(data, dtype=np.int)

        feature_dim = self.vocab.size
        if self.vocab.use_special_tokens:
            feature_dim -= len(SPECIAL_TOKENS)
        X = sp.csr_matrix((data, indices, indptr),
                          shape=(len(sequences), feature_dim),
                          dtype=np.int64)
        X.sort_indices()
        return X

    def fit(self, raw_documents: List[str]) -> 'CountVectorizer':
        """Build a vocabulary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents: List[str]) -> (List[List[str]], sp.csr_matrix):
        """Build the vocabulary and return term-document matrix.

        Parameters
        ----------
        raw_documents : List[str]

        Returns
        -------
        (sequences, X) :
            sequences: List[List[str]
                Tokenized sequences of raw_documents
            X: array, [n_samples, n_features]
                Document-term matrix.
        """
        sequences = self.tokenizer.batch_tokenize(raw_documents)

        fixed_vocab = self.vocab is not None
        if self.vocab is None:
            self.vocab = Vocabulary.from_sequences(sequences, min_freq=self.min_freq)

        X = self._count(sequences)
        if self.binary:
            X.data.fill(1)

        if not fixed_vocab:
            n_docs = X.shape[0]
            max_doc_count = (self.max_doc_freq
                             if isinstance(self.max_doc_freq, int)
                             else self.max_doc_freq * n_docs)
            X = self._limit_features(X, max_doc_count)

        return sequences, X

    def transform(self, raw_documents: List[str]) -> (List[List[str]], sp.csr_matrix):
        """Transform documents to document-term matrix.

        Parameters
        ----------
        raw_documents : List[str]

        Returns
        -------
        (sequences, X) :
            sequences: List[List[str]
                Tokenized sequences of raw_documents.
            X: array, [n_samples, n_features]
                Document-term matrix.
        """
        sequences = self.tokenizer.batch_tokenize(raw_documents)
        X = self._count(sequences)
        if self.binary:
            X.data.fill(1)
        return sequences, X


class TextModule(FeatureModule):
    """Text module

    Parameters
    ----------
    corpus: List[str], default = None
        List of user/item texts that the indices are aligned with `ids`.

    ids: List, default = None
        List of user/item ids that the indices are aligned with `corpus`.
        If None, the indices of provided `corpus` will be used as `ids`.

    tokenizer: Tokenizer, optional, default = None
        Tokenizer for text splitting. If None, the BaseTokenizer will be used.

    vocab: Vocabulary, optional, default = None
        Vocabulary of tokens. It contains mapping between tokens to their
        integer ids and vice versa.

    max_vocab: int, optional, default = None
        The maximum size of the vocabulary.
        If vocab is provided, this will be ignored.

    max_doc_freq: Union[float, int] = 1.0
        The maximum frequency of tokens appearing in documents to be excluded from vocabulary.
        If float, the value represents a proportion of documents, int for absolute counts.
        If `vocab` is not None, this will be ignored.

    min_freq: int, default = 1
        The minimum frequency of tokens to be included into vocabulary.
        If `vocab` is not None, this will be ignored.

    """

    def __init__(self,
                 corpus: List[str] = None,
                 ids: List = None,
                 tokenizer: Tokenizer = None,
                 vocab: Vocabulary = None,
                 max_vocab: int = None,
                 max_doc_freq: Union[float, int] = 1.0,
                 min_freq: int = 1,
                 **kwargs):
        super().__init__(ids=ids, **kwargs)
        self.corpus = corpus
        self.tokenizer = BaseTokenizer() if tokenizer is None else tokenizer
        self.vocab = vocab
        self.max_vocab = max_vocab
        self.max_doc_freq = max_doc_freq
        self.min_freq = min_freq
        self.sequences = None
        self.count_matrix = None

    def _swap_text(self, id_map: Dict):
        for old_idx, raw_id in enumerate(self._ids.copy()):
            new_idx = id_map.get(raw_id, None)
            if new_idx is None:
                continue
            assert new_idx < len(self.corpus)
            self.corpus[old_idx], self.corpus[new_idx] = self.corpus[new_idx], self.corpus[old_idx]
            self._ids[old_idx], self._ids[new_idx] = self._ids[new_idx], self._ids[old_idx]

    def _build_text(self, id_map: Dict):
        """Build the text based on provided global id map
        """
        if self.corpus is None:
            return

        if (self._ids is not None) and (id_map is not None):
            self._swap_text(id_map)

        vectorizer = CountVectorizer(tokenizer=self.tokenizer, vocab=self.vocab,
                                     max_doc_freq=self.max_doc_freq, min_freq=self.min_freq,
                                     max_features=self.max_vocab, binary=False)
        self.sequences, self.count_matrix = vectorizer.fit_transform(self.corpus)
        self.vocab = Vocabulary(vectorizer.vocab.idx2tok, use_special_tokens=True)
        # Map tokens into integer ids
        for i, seq in enumerate(self.sequences):
            self.sequences[i] = self.vocab.to_idx(seq)

    def build(self, id_map=None):
        """Build the model based on provided list of ordered ids
        """
        super().build(id_map)
        self._build_text(id_map)
        return self

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

    @fallback_feature
    def batch_bow(self, batch_ids, binary=False, keep_sparse=False):
        """Return matrix of bag-of-words corresponding to provided batch_ids

        Parameters
        ----------
        batch_ids: array
            An array of ids to retrieve the corresponding features.

        binary: bool, default = False
            If `True`, the feature values will be converted into (0 or 1).

        keep_sparse: bool, default = False
            If `True`, the return feature matrix will be a `scipy.sparse.csr_matrix`.
            Otherwise, it will be a dense matrix.

        """
        if self.count_matrix is None:
            raise ValueError('self.count_matrix is required but None!')

        bow_mat = self.count_matrix[batch_ids]
        if binary:
            bow_mat.data.fill(1)

        if keep_sparse:
            return bow_mat

        return bow_mat.A

    def batch_tfidf(self, batch_ids):
        """Return matrix of TF-IDF features corresponding to provided batch_ids
        """
        raise NotImplementedError
