import unittest

import torch
from cornac.data.transformer_text import TransformersTextModality
from sentence_transformers import util



class TestTransformersTextModality(unittest.TestCase):

    def setUp(self):
        corpus = ["I like you very much.", "I like you so much"]
        self.ids = [0, 1]
        self.modality = TransformersTextModality(corpus=corpus, ids=self.ids, preencode=True)

    def test_batch_encode(self):
        encoded_batch = self.modality.batch_encode(self.ids)

        assert encoded_batch.shape[0] == 2
        assert isinstance(encoded_batch, torch.Tensor)

    def test_batch_encode_similarity(self):
        encoded_batch = self.modality.batch_encode(self.ids)
        similarity = util.cos_sim(encoded_batch[0], encoded_batch[1])
        assert similarity > 0.9
