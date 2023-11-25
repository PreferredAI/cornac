import unittest

import numpy as np
import unittest
import torch

from cornac.data.bert_text import BertTextModality
from sentence_transformers import util

from cornac.models.dmrl.dmrl import DMRL

class TestBertTextModality(unittest.TestCase):

    def setUp(self):
        corpus = ["I like you very much.", "I like you so much"]
        self.ids = [0, 1]
        self.modality = BertTextModality(corpus=corpus, ids=self.ids)

    def test_batch_encode(self):
        encoded_batch = self.modality.batch_encode(self.ids)

        assert encoded_batch.shape[0] == 2
        assert isinstance(encoded_batch, torch.Tensor)

    def test_batch_encode_similarity(self):
        encoded_batch = self.modality.batch_encode(self.ids)
        similarity = util.cos_sim(encoded_batch[0], encoded_batch[1])
        assert similarity > 0.9


class TestDMRL(unittest.TestCase):

    def setUp(self):
        corpus = ["this is an amazing book. it is so great", "This bike was ok"]
        self.item_ids = [0, 1]
        self.feedback = [(0, 0, 1), (0, 1, 1), (1, 1, 1)]
        num_items = 2
        num_users = 2
        embedding_dim = 100
        bert_text_dim = 384
        self.modality = BertTextModality(corpus=corpus, ids=self.item_ids)

        self.input_tensor_u_ids = torch.tensor([i[0] for i in self.feedback])
        self.input_tensor_i_ids = torch.tensor([i[1] for i in self.feedback])
        self.dmrl = DMRL(num_users, num_items, embedding_dim, bert_text_dim)
        self.item_text_embeddimgs = self.modality.batch_encode(self.input_tensor_i_ids)

    def test_forward_pass(self):
        # Create a random input tensor
        input = torch.randn(10, 10)

        # Forward pass through the network
        output = self.dmrl(self.input_tensor_u_ids, self.input_tensor_i_ids, self.item_text_embeddimgs)

        # Check that the output tensor has the correct size
        self.assertEqual(output.size(), (10, 10))

    def test_backward_pass(self):
        # Create a random input tensor
        input = torch.randn(10, 10)

        # Target output tensor
        target = torch.randn(10, 10)

        # Forward pass through the network
        output = self.mlp(input)

        # Compute the loss
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass through the network
        loss.backward()

        # Check that the gradients of the network parameters are not zero
        for param in self.mlp.parameters():
            self.assertNotEqual(param.grad, torch.zeros_like(param))
