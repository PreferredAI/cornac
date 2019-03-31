# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest

from cornac.data import TextModule
from cornac.data.text import BaseTokenizer
from cornac.data.text import Vocabulary
from cornac.data.text import CountVectorizer
from cornac.data.text import SPECIAL_TOKENS, DEFAULT_PRE_RULES
from collections import defaultdict
import numpy as np
import numpy.testing as npt


class TestBaseTokenizer(unittest.TestCase):

    def setUp(self):
        self.tok = BaseTokenizer()

    def test_init(self):
        self.assertEqual(self.tok.sep, ' ')

    def test_tokenize(self):
        tokens = self.tok.tokenize('a b c')
        self.assertListEqual(tokens, ['a', 'b', 'c'])

    def test_batch_tokenize(self):
        token_list = self.tok.batch_tokenize(['a b c',
                                              'd e f'])
        self.assertListEqual(token_list, [['a', 'b', 'c'],
                                          ['d', 'e', 'f']])

    def test_default_rules(self):
        tok = BaseTokenizer(pre_rules=DEFAULT_PRE_RULES)
        token_list = tok.tokenize('<t>a</t> B |{ C ]?&$  d123 E')
        self.assertListEqual(token_list, ['a', 'b', 'c', 'd', 'e'])

    def test_stopwords(self):
        text = 'this is a nice house'

        tok = BaseTokenizer(stop_words='english')
        self.assertListEqual(tok.tokenize(text), ['nice', 'house'])

        tok = BaseTokenizer(stop_words=['is', 'a'])
        self.assertListEqual(tok.tokenize(text), ['this', 'nice', 'house'])

        try:
            BaseTokenizer(stop_words='vietnamese')
        except ValueError:
            assert True


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.tokens = ['a', 'b', 'c']
        self.vocab = Vocabulary(self.tokens)
        (a, b, c) = (self.vocab.tok2idx[tok] for tok in self.tokens[-3:])
        self.tok_seq = ['a', 'a', 'b', 'c']
        self.idx_seq = [a, a, b, c]

    def test_init(self):
        self.assertEqual(self.vocab.size, 3)
        self.assertListEqual(self.vocab.idx2tok, ['a', 'b', 'c'])

        tok2idx = defaultdict()
        for tok in self.tokens:
            tok2idx.setdefault(tok, len(tok2idx))
        self.assertDictEqual(self.vocab.tok2idx, tok2idx)

    def test_to_idx(self):
        self.assertEqual(self.vocab.to_idx(self.tok_seq), self.idx_seq)

    def test_to_text(self):
        self.assertEqual(self.vocab.to_text(self.idx_seq), ' '.join(self.tok_seq))
        self.assertEqual(self.vocab.to_text(self.idx_seq, sep=None), self.tok_seq)

    def test_save(self):
        self.vocab.save('tests/vocab.pkl')
        loaded_vocab = Vocabulary.load('tests/vocab.pkl')
        self.assertListEqual(self.vocab.idx2tok, loaded_vocab.idx2tok)

    def test_from_tokens(self):
        from_tokens_vocab = Vocabulary.from_tokens(self.tokens)
        self.assertCountEqual(self.vocab.idx2tok, from_tokens_vocab.idx2tok)

    def test_from_sequences(self):
        from_sequences_vocab = Vocabulary.from_sequences([self.tokens])
        self.assertCountEqual(self.vocab.idx2tok, from_sequences_vocab.idx2tok)

    def test_special_tokens(self):
        tokens = ['a', 'b', 'c', SPECIAL_TOKENS[1]]
        vocab = Vocabulary(tokens, use_special_tokens=True)

        self.assertEqual(vocab.size, len(SPECIAL_TOKENS) + 3)
        self.assertListEqual(vocab.idx2tok, SPECIAL_TOKENS + ['a', 'b', 'c'])

        tok2idx = defaultdict()
        for tok in SPECIAL_TOKENS + tokens:
            tok2idx.setdefault(tok, len(tok2idx))
        self.assertDictEqual(vocab.tok2idx, tok2idx)


class TestCountVectorizer(unittest.TestCase):

    def setUp(self):
        self.docs = ['a b c',
                     'b c d d',
                     'c b e c f']

    def test_arguments(self):
        try:
            CountVectorizer(max_doc_freq=-1)
        except ValueError:
            assert True

        try:
            CountVectorizer(max_features=-1)
        except ValueError:
            assert True

    def test_bad_freq_arguments(self):
        vectorizer = CountVectorizer(max_doc_freq=2, min_freq=3)
        try:
            vectorizer.fit(self.docs)
        except ValueError:
            assert True

    def test_transform(self):
        vectorizer = CountVectorizer(max_doc_freq=2, min_freq=1, max_features=1)
        vectorizer.fit(self.docs)
        sequences, X = vectorizer.transform(self.docs)
        npt.assert_array_equal(X.A, np.asarray([[0], [2], [0]]))

        vectorizer.binary = True
        _, X1 = vectorizer.fit_transform(self.docs)
        _, X2 = vectorizer.transform(self.docs)
        npt.assert_array_equal(X1.A, X2.A)

    def test_with_special_tokens(self):
        vectorizer = CountVectorizer(max_doc_freq=2, min_freq=1, max_features=1)
        vectorizer.fit(self.docs)

        new_vocab = Vocabulary(vectorizer.vocab.idx2tok, use_special_tokens=True)
        vectorizer.vocab = new_vocab

        sequences, X = vectorizer.transform(self.docs)
        npt.assert_array_equal(X.A, np.asarray([[0], [2], [0]]))


class TestTextModule(unittest.TestCase):

    def setUp(self):
        self.tokens = ['a', 'b', 'c', 'd', 'e', 'f']
        corpus = ['a b c', 'b c d d', 'c b e c f']
        ids = ['u1', 'u2', 'u3']
        # frequency ranking: c > b > d > a > e > f
        self.module = TextModule(corpus=corpus, ids=ids, max_vocab=6)
        self.module.build({'u1': 0, 'u2': 1, 'u3': 2})
        self.token_ids = (self.module.vocab.tok2idx[tok] for tok in self.tokens)

    def test_init(self):
        self.assertCountEqual(self.module.vocab.idx2tok,
                              SPECIAL_TOKENS + self.tokens)

    def test_build(self):
        TextModule().build()
        TextModule(corpus=['abc']).build()
        TextModule(corpus=['abc']).build({'b': 0})
        TextModule(corpus=['abc'], ids=['a']).build({'b': 0})

    def test_sequences(self):
        (a, b, c, d, e, f) = self.token_ids

        self.assertListEqual(self.module.sequences,
                             [[a, b, c],
                              [b, c, d, d],
                              [c, b, e, c, f]])

    def test_batch_seq(self):
        (a, b, c, d, e, f) = self.token_ids

        batch_seqs = self.module.batch_seq([2, 1])
        self.assertEqual((2, 5), batch_seqs.shape)
        npt.assert_array_equal(batch_seqs,
                               np.asarray([[c, b, e, c, f],
                                           [b, c, d, d, 0]]))

        batch_seqs = self.module.batch_seq([0, 2], max_length=4)
        self.assertEqual((2, 4), batch_seqs.shape)
        npt.assert_array_equal(batch_seqs,
                               np.asarray([[a, b, c, 0],
                                           [c, b, e, c]]))

        self.module.sequences = None
        try:
            self.module.batch_seq([0])
        except ValueError:
            assert True

    def test_count_matrix(self):
        (a, b, c, d, e, f) = self.token_ids
        shift = len(SPECIAL_TOKENS)
        expected_counts = np.zeros_like(self.module.count_matrix.A)
        expected_counts[0, a - shift] = 1
        expected_counts[0, b - shift] = 1
        expected_counts[0, c - shift] = 1
        expected_counts[1, b - shift] = 1
        expected_counts[1, c - shift] = 1
        expected_counts[1, d - shift] = 2
        expected_counts[2, b - shift] = 1
        expected_counts[2, c - shift] = 2
        expected_counts[2, e - shift] = 1
        expected_counts[2, f - shift] = 1
        npt.assert_array_equal(self.module.count_matrix.A, expected_counts)

    def test_batch_bow(self):
        (a, b, c, d, e, f) = self.token_ids
        shift = len(SPECIAL_TOKENS)

        batch_bows = self.module.batch_bow([2, 1])
        self.assertEqual((2, self.module.max_vocab), batch_bows.shape)
        expected_bows = np.zeros_like(batch_bows)
        expected_bows[0, b - shift] = 1
        expected_bows[0, c - shift] = 2
        expected_bows[0, e - shift] = 1
        expected_bows[0, f - shift] = 1
        expected_bows[1, b - shift] = 1
        expected_bows[1, c - shift] = 1
        expected_bows[1, d - shift] = 2
        npt.assert_array_equal(batch_bows, expected_bows)

        batch_bows = self.module.batch_bow([0, 2], binary=True, keep_sparse=True)
        self.assertEqual((2, 6), batch_bows.shape)
        expected_bows = np.zeros_like(batch_bows.A)
        expected_bows[0, np.asarray([a, b, c]) - shift] = 1
        expected_bows[1, np.asarray([b, c, e, f]) - shift] = 1
        npt.assert_array_equal(batch_bows.A, expected_bows)

        self.module.count_matrix = None
        try:
            self.module.batch_bow([0])
        except ValueError:
            assert True

    def test_batch_bow_fallback(self):
        module = TextModule(features=np.asarray([[3, 2, 1], [4, 5, 6]]),
                            ids=['a', 'b'])
        module.build()
        npt.assert_array_equal(np.asarray([[3, 2, 1]]), module.batch_bow(batch_ids=[0]))


if __name__ == '__main__':
    unittest.main()
