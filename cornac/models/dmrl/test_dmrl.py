
from copy import deepcopy
import unittest

import torch
from torch.utils.data import DataLoader
from cornac.data.bert_text import BertTextModality
from cornac.data.dataset import Dataset
from cornac.data.reader import Reader
from cornac.datasets import citeulike
import os

from cornac.models.dmrl.dmrl import DMRLLoss, DMRLModel
from cornac.models.dmrl.pwlearning_sampler import PWLearningSampler


class TestDMRL(unittest.TestCase):

    def setUp(self):
        # initialize sampler
        self.num_neg = 4
        docs, item_ids = citeulike.load_text()
        feedback = citeulike.load_feedback(reader=Reader(item_set=item_ids))
        cornac_dataset = Dataset.build(
            data=feedback)
        self.sampler = PWLearningSampler(cornac_dataset, num_neg=self.num_neg)
        cornac_dataset = Dataset.build(
            data=feedback)

        self.item_ids = item_ids
        embedding_dim = 100
        bert_text_dim = 384
        self.modality = BertTextModality(corpus=docs, ids=self.item_ids)
        self.modality.build(cornac_dataset.iid_map)
        self.dmrl = DMRLModel(cornac_dataset.num_users, cornac_dataset.num_items, embedding_dim, bert_text_dim, self.num_neg)
        self.loss_func = DMRLLoss(decay_c=1e-3, num_factors=self.dmrl.num_factors, num_neg=self.num_neg)
        self.sampler = PWLearningSampler(cornac_dataset, num_neg=self.num_neg)

        batch_size = 32
        path = 'temp/input_tensor.pt'
        try:
            self.batch = torch.load(path)
        except FileNotFoundError:
            dataloader = DataLoader(self.sampler, batch_size=batch_size, num_workers=4, shuffle=True, prefetch_factor=3)
            generator_data_loader = iter(dataloader)
            
            #col 0 is users, col1 is pos items col 2 through num_neg is negative items
            self.batch = next(generator_data_loader)
            os.makedirs("temp", exist_ok = True) 
            torch.save(self.batch, path)

        # get the encodings for the items (positive in col1, neg in 2 through last col)
        shape = self.batch[:, 1:].shape
        all_items = self.batch[:, 1:].flatten()
        self.item_text_embeddimgs = self.modality.batch_encode(all_items)
        self.item_text_embeddimgs = self.item_text_embeddimgs.reshape((*shape, self.modality.output_dim))

    def test_forward_pass(self):
        # Forward pass through the network
        output = self.dmrl(self.batch, self.item_text_embeddimgs)

        # Check that the output tensor has the correct size
        self.assertEqual(output.size(), (10, 10))

    def test_backward_pass(self):
        # forward pass through network
        embedding_factor_lists, ratings = self.dmrl(self.batch, self.item_text_embeddimgs)
        
        # Compute the loss
        loss = self.loss_func(embedding_factor_lists, ratings)

        # Backward pass through the network
        loss.backward()

        # Check that the gradients of the network parameters are not zero
        for param in self.dmrl.parameters():
            assert  torch.sum(param.grad) != 0

    def test_one_training_epoch_one_batch(self):
        optimizer = torch.optim.Adam(self.dmrl.parameters())
        old_params = deepcopy(self.dmrl.state_dict())
        optimizer.zero_grad(set_to_none=False)
        # forward pass through network
        embedding_factor_lists, ratings = self.dmrl(self.batch, self.item_text_embeddimgs)        
        # Compute the loss
        loss = self.loss_func(embedding_factor_lists, ratings)

        # Backward pass through the network to compute the gradients
        loss.backward()

        # Update the network parameters
        optimizer.step()

        # Check that the parameters of the network have been updated
        for key, tensor in self.dmrl.state_dict().items():
            assert not torch.eq(old_params[key], tensor).all()
