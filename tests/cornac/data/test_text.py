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

import unittest
from collections import defaultdict

import numpy as np
import numpy.testing as npt

from cornac.data import TextModality, ReviewModality
from cornac.data.text import (
    SPECIAL_TOKENS,
    DEFAULT_PRE_RULES,
    BaseTokenizer,
    Vocabulary,
    CountVectorizer,
    TfidfVectorizer,
)


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
        vectorizer = CountVectorizer(max_doc_freq=2, min_doc_freq=3)
        try:
            vectorizer.fit(self.docs)
        except ValueError:
            assert True

    def test_transform(self):
        vectorizer = CountVectorizer(max_doc_freq=2, min_doc_freq=1, max_features=1)
        vectorizer.fit(self.docs)
        sequences, X = vectorizer.transform(self.docs)
        npt.assert_array_equal(X.toarray(), np.asarray([[0], [2], [0]]))

        vectorizer.binary = True
        _, X1 = vectorizer.fit_transform(self.docs)
        _, X2 = vectorizer.transform(self.docs)
        npt.assert_array_equal(X1.toarray(), X2.toarray())

    def test_with_special_tokens(self):
        vectorizer = CountVectorizer(max_doc_freq=2, min_doc_freq=1, max_features=1)
        vectorizer.fit(self.docs)

        new_vocab = Vocabulary(vectorizer.vocab.idx2tok, use_special_tokens=True)
        vectorizer.vocab = new_vocab

        sequences, X = vectorizer.transform(self.docs)
        npt.assert_array_equal(X.toarray(), np.asarray([[0], [2], [0]]))


class TestTfidfVectorizer(unittest.TestCase):

    def setUp(self):
        self.docs = ['this is a sample',
                     'this is another example']

    def test_arguments(self):
        try:
            TfidfVectorizer(max_doc_freq=-1)
        except ValueError:
            assert True

        try:
            TfidfVectorizer(max_features=-1)
        except ValueError:
            assert True

    def test_bad_freq_arguments(self):
        vectorizer = TfidfVectorizer(max_doc_freq=2, min_doc_freq=3)
        try:
            vectorizer.fit(self.docs)
        except ValueError:
            assert True

    def test_transform(self):
        vectorizer = TfidfVectorizer(norm=None)
        vectorizer.fit(self.docs)

        tok2idx = vectorizer.vocab.tok2idx
        idf = vectorizer.idf
        print(vectorizer.vocab.idx2tok)

        self.assertEqual(idf[tok2idx['this'], tok2idx['this']], 1)
        self.assertEqual(idf[tok2idx['a'], tok2idx['a']], np.log(3 / 2) + 1)

        X = vectorizer.transform(self.docs).toarray()
        npt.assert_array_equal(X[:, tok2idx['this']],
                               np.asarray([1., 1.]))
        npt.assert_array_equal(X[:, tok2idx['a']],
                               np.asarray([1., 0.]) * (np.log(3 / 2) + 1))

        vectorizer.binary = True
        vectorizer.sublinear_tf = True
        X1 = vectorizer.fit_transform(self.docs)
        X2 = vectorizer.transform(self.docs)
        npt.assert_array_equal(X1.toarray(), X2.toarray())


class TestTextModality(unittest.TestCase):

    def setUp(self):
        self.tokens = ['a', 'b', 'c', 'd', 'e', 'f']
        corpus = ['a b c', 'b c d d', 'c b e c f']
        ids = ['u1', 'u2', 'u3']
        # frequency ranking: c > b > d > a > e > f
        self.modality = TextModality(corpus=corpus, ids=ids, max_vocab=6)
        self.modality.build({'u1': 0, 'u2': 1, 'u3': 2})
        self.token_ids = (self.modality.vocab.tok2idx[tok] for tok in self.tokens)

    def test_init(self):
        self.assertCountEqual(self.modality.vocab.idx2tok,
                              SPECIAL_TOKENS + self.tokens)

    def test_build(self):
        TextModality().build()
        TextModality(corpus=['abc']).build()
        TextModality(corpus=['abc']).build({'b': 0})
        TextModality(corpus=['abc'], ids=['a']).build({'b': 0})

    def test_sequences(self):
        (a, b, c, d, e, f) = self.token_ids

        self.assertListEqual(self.modality.sequences,
                             [[a, b, c],
                              [b, c, d, d],
                              [c, b, e, c, f]])

    def test_batch_seq(self):
        (a, b, c, d, e, f) = self.token_ids

        batch_seqs = self.modality.batch_seq([2, 1])
        self.assertEqual((2, 5), batch_seqs.shape)
        npt.assert_array_equal(batch_seqs,
                               np.asarray([[c, b, e, c, f],
                                           [b, c, d, d, 0]]))

        batch_seqs = self.modality.batch_seq([0, 2], max_length=4)
        self.assertEqual((2, 4), batch_seqs.shape)
        npt.assert_array_equal(batch_seqs,
                               np.asarray([[a, b, c, 0],
                                           [c, b, e, c]]))

        self.modality.sequences = None
        try:
            self.modality.batch_seq([0])
        except ValueError:
            assert True

    def test_count_matrix(self):
        (a, b, c, d, e, f) = self.token_ids
        shift = len(SPECIAL_TOKENS)
        expected_counts = np.zeros_like(self.modality.count_matrix.toarray())
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
        npt.assert_array_equal(self.modality.count_matrix.toarray(), expected_counts)

    def test_batch_bow(self):
        (a, b, c, d, e, f) = self.token_ids
        shift = len(SPECIAL_TOKENS)

        batch_bows = self.modality.batch_bow([2, 1])
        self.assertEqual((2, self.modality.max_vocab), batch_bows.shape)
        expected_bows = np.zeros_like(batch_bows)
        expected_bows[0, b - shift] = 1
        expected_bows[0, c - shift] = 2
        expected_bows[0, e - shift] = 1
        expected_bows[0, f - shift] = 1
        expected_bows[1, b - shift] = 1
        expected_bows[1, c - shift] = 1
        expected_bows[1, d - shift] = 2
        npt.assert_array_equal(batch_bows, expected_bows)

        batch_bows = self.modality.batch_bow([0, 2], binary=True, keep_sparse=True)
        self.assertEqual((2, 6), batch_bows.shape)
        expected_bows = np.zeros_like(batch_bows.toarray())
        expected_bows[0, np.asarray([a, b, c]) - shift] = 1
        expected_bows[1, np.asarray([b, c, e, f]) - shift] = 1
        npt.assert_array_equal(batch_bows.toarray(), expected_bows)

        self.modality.count_matrix = None
        try:
            self.modality.batch_bow([0])
        except ValueError:
            assert True

    def test_batch_bow_fallback(self):
        modality = TextModality(features=np.asarray([[3, 2, 1], [4, 5, 6]]),
                                ids=['a', 'b'])
        modality.build()
        npt.assert_array_equal(np.asarray([[3, 2, 1]]), modality.batch_bow(batch_ids=[0]))

    def test_batch_tfidf(self):
        batch_tfidf = self.modality.batch_tfidf([2, 1])
        self.assertEqual((2, self.modality.max_vocab), batch_tfidf.shape)

    def test_tfidf_params(self):
        corpus = ['a b c', 'b c d d', 'c b e c f']
        ids = ['u1', 'u2', 'u3']

        modality = TextModality(corpus=corpus, ids=ids, max_vocab=6,
                                tfidf_params={
                                    'binary': False,
                                    'norm': 'l2',
                                    'use_idf': True,
                                    'smooth_idf': True,
                                    'sublinear_tf': False
                                }).build({'u1': 0, 'u2': 1, 'u3': 2})
        npt.assert_array_equal(modality.batch_tfidf([1]),
                               self.modality.batch_tfidf([1]))

        for k, v in {
            'binary': True,
            'norm': 'l1',
            'use_idf': False,
            'smooth_idf': False,
            'sublinear_tf': True
        }.items():
            modality = TextModality(corpus=corpus, ids=ids, max_vocab=6,
                                    tfidf_params={k: v})
            modality.build({'u1': 0, 'u2': 1, 'u3': 2})
            self.assertFalse(np.array_equal(modality.batch_tfidf([1]),
                                            self.modality.batch_tfidf([1])))

class TestReviewModality(unittest.TestCase):
    def setUp(self):
        self.tokens = ['a', 'b', 'c', 'd', 'e', 'f']
        self.review_data = [
            ('76', '93', 'a b c'),
            ('76', '257', 'b c c')
        ]
        self.uid_map = {'76': 0, '642': 1, '930': 2}
        self.iid_map = {'93': 0, '257': 1, '705': 2}
        self.dok_matrix = np.array([
            [4, 4, 0],
            [0, 4, 0],
            [0, 4, 0],
        ], dtype='int')

    def test_init(self):
        try:
            ReviewModality(group_by='something')
        except ValueError:
            assert True

    def test_build(self):
        ReviewModality(data=[]).build(uid_map=self.uid_map, iid_map=self.iid_map, dok_matrix=self.dok_matrix)
        ReviewModality(data=self.review_data).build(uid_map=self.uid_map, iid_map=self.iid_map, dok_matrix=self.dok_matrix)
        ReviewModality(data=self.review_data, group_by='user').build(uid_map=self.uid_map, iid_map=self.iid_map, dok_matrix=self.dok_matrix)
        ReviewModality(data=self.review_data, group_by='item').build(uid_map=self.uid_map, iid_map=self.iid_map, dok_matrix=self.dok_matrix)
        try:
            ReviewModality().build()
        except ValueError:
            assert True

if __name__ == '__main__':
    unittest.main()
