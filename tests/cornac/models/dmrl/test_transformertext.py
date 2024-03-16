# add a checker to make sure all requirements needed in the imports here are really present.
# if they are missing skip the respective test
# If a user wants to un these please run: pip install -r cornac/models/dmrl/requirements.txt

import unittest

try:
    import torch
    from sentence_transformers import util
    from cornac.models.dmrl.transformer_text import TransformersTextModality

    run_dmrl_test_funcs = True

except ImportError:
    run_dmrl_test_funcs = False


def skip_test_in_case_of_missing_reqs(test_func):
    test_func.__test__ = (
        run_dmrl_test_funcs  # Mark the test function as (non-)discoverable by unittest
    )
    return test_func


class TestTransformersTextModality(unittest.TestCase):
    @skip_test_in_case_of_missing_reqs
    def setUp(self):
        self.corpus = ["I like you very much.", "I like you so much"]
        self.ids = [0, 1]
        self.modality = TransformersTextModality(
            corpus=self.corpus, ids=self.ids, preencode=True
        )

    @skip_test_in_case_of_missing_reqs
    def test_batch_encode(self):
        encoded_batch = self.modality.batch_encode(self.ids)

        assert encoded_batch.shape[0] == 2
        assert isinstance(encoded_batch, torch.Tensor)

    @skip_test_in_case_of_missing_reqs
    def test_preencode_entire_corpus(self):
        self.modality.preencode_entire_corpus()
        assert self.modality.preencoded

        assert torch.load("temp/encoded_corpus_ids.pt") == self.ids
        assert torch.load("temp/encoded_corpus.pt").shape[0] == len(self.corpus)

    @skip_test_in_case_of_missing_reqs
    def test_batch_encode_similarity(self):
        encoded_batch = self.modality.batch_encode(self.ids)
        similarity = util.cos_sim(encoded_batch[0], encoded_batch[1])
        assert similarity > 0.9
