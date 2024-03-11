import unittest
from torch.utils.data import DataLoader
from cornac.data.dataset import Dataset
from cornac.data.reader import Reader
from cornac.datasets import citeulike
from cornac.models.dmrl.pwlearning_sampler import PWLearningSampler


class TestPWLearningSampler(unittest.TestCase):
    """
    Tests that the PW Sampler returns the desired batch of data.
    """
    def setUp(self):
        self.num_neg = 4
        _, item_ids = citeulike.load_text()
        feedback = citeulike.load_feedback(reader=Reader(item_set=item_ids))
        cornac_dataset = Dataset.build(
            data=feedback)
        self.sampler = PWLearningSampler(cornac_dataset, num_neg=self.num_neg)

    def test_get_batch_multiprocessed(self):
        """
        Tests multiprocessed loading via Torch Datalodaer
        """
        batch_size = 32
        dataloader = DataLoader(self.sampler, batch_size=batch_size, num_workers=3, shuffle=True, prefetch_factor=3)
        generator_data_loader = iter(dataloader)
        batch = next(generator_data_loader)
        assert batch.shape == (batch_size, 2+self.num_neg)

    def test_correctness(self):
        """
        Tests the correctness of the PWLearningSampler by asserting that the
        correct positive and negative user-item pairs are returned.
        """
        batch_size = 32
        dataloader = DataLoader(self.sampler, batch_size=batch_size, num_workers=0, shuffle=True, prefetch_factor=None)
        generator_data_loader = iter(dataloader)
        batch = next(generator_data_loader).numpy()
        assert batch.shape == (batch_size, 2+self.num_neg)
        for i in range(batch_size):
            user = batch[i, 0]
            pos_item = batch[i, 1]
            neg_items = batch[i, 2:]
            assert pos_item in self.sampler.data.csr_matrix[user].nonzero()[1]
            for neg_item in neg_items:
                assert neg_item not in self.sampler.data.csr_matrix[user].nonzero()[1]

    def test_full_epoch_sampler(self):
        """
        Tests speed of loader for full epoch
        """
        batch_size = 32
        dataloader = DataLoader(self.sampler, batch_size=batch_size, num_workers=0, shuffle=True, prefetch_factor=None)
        i = 0
        for _ in dataloader:
            i += 1
        assert i == self.sampler.user_array.shape[0] // batch_size + 1