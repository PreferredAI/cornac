# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

import unittest

from cornac.data import TextModule
from cornac.data.text import BaseTokenizer
from cornac.data.text import Vocabulary
from cornac.data.text import SPECIAL_TOKENS, DEFAULT_PRE_RULES, DEFAULT_POST_RULES
from collections import OrderedDict, defaultdict
import numpy as np


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
        tok = BaseTokenizer(pre_rules=DEFAULT_PRE_RULES, post_rules=DEFAULT_POST_RULES)
        token_list = tok.tokenize('a B  C   d E')
        self.assertListEqual(token_list, ['a', 'b', 'c', 'd', 'e'])


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.tokens = ['a', 'b', 'c']
        self.vocab = Vocabulary(self.tokens)
        (a, b, c) = (self.vocab.tok2idx[tok] for tok in self.tokens[-3:])
        self.tok_seq = ['a', 'a', 'b', 'c']
        self.idx_seq = [a, a, b, c]

    def test_init(self):
        self.assertEqual(self.vocab.size, len(SPECIAL_TOKENS) + 3)
        self.assertListEqual(self.vocab.idx2tok, SPECIAL_TOKENS + ['a', 'b', 'c'])

        tok2idx = defaultdict()
        for tok in SPECIAL_TOKENS + self.tokens:
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


class TestTextModule(unittest.TestCase):

    def setUp(self):
        self.tokens = ['a', 'b', 'c', 'd', 'e', 'f']
        self.id_text = {'u1': 'a b c',
                        'u2': 'b c d d',
                        'u3': 'c b e c f'}

        self.module = TextModule(self.id_text)
        self.id_map = OrderedDict({'u1': 0, 'u2': 1, 'u3': 2})
        self.module.build(self.id_map)
        self.token_ids = (self.module.vocab.tok2idx[tok] for tok in self.tokens)

    def test_init(self):
        self.assertCountEqual(self.module.vocab.idx2tok,
                              SPECIAL_TOKENS + self.tokens)

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
        np.testing.assert_array_equal(batch_seqs,
                                      np.asarray([[c, b, e, c, f],
                                                  [b, c, d, d, 0]]))

        batch_seqs = self.module.batch_seq([0, 2], max_length=4)
        self.assertEqual((2, 4), batch_seqs.shape)
        np.testing.assert_array_equal(batch_seqs,
                                      np.asarray([[a, b, c, 0],
                                                  [c, b, e, c]]))

        self.module.sequences = None
        try:
            self.module.batch_seq([0])
        except ValueError:
            assert True


if __name__ == '__main__':
    unittest.main()
