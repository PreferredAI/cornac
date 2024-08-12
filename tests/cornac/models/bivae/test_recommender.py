import unittest

from cornac.data import Dataset, Reader
from cornac.models import BiVAECF


class TestRecommender(unittest.TestCase):
    def setUp(self):
        self.data = Reader().read("./tests/data.txt")

    def test_run(self):
        bivae = BiVAECF(k=1, seed=123)
        dataset = Dataset.from_uir(self.data)
        # Assert runs without error
        bivae.fit(dataset)
