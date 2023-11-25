import unittest
from torch.utils.data import DataLoader
from cornac.data.dataset import Dataset
from cornac.data.reader import Reader
from cornac.datasets import citeulike
from cornac.models.dmrl.pwlearning_sampler import PWLearningSampler


class TestPWLearningSampler(unittest.TestCase):

    def setUp(self):
        self.num_neg = 2
        docs, item_ids = citeulike.load_text()
        feedback = citeulike.load_feedback(reader=Reader(item_set=item_ids))
        cornac_dataset = Dataset.build(
            data=feedback)
        self.sampler = PWLearningSampler(cornac_dataset, num_neg=self.num_neg)

    def test_get_batch(self):
        """
        Tests multiprocessed loading via Torch Datalodaer
        """
        batch_size = 32
        dataloader = DataLoader(self.sampler, batch_size=batch_size, num_workers=4, shuffle=True, prefetch_factor=3)
        generator_data_loader = iter(dataloader)
        batch = next(generator_data_loader)
        assert batch.shape == (batch_size, 2+self.num_neg)
