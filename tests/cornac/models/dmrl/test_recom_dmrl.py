# Tests for saving and loading a trained DMRL model.
# They are skipped if the DMRL requirements or tensorboard are missing. To run them:
# pip install -r cornac/models/dmrl/requirements.txt tensorboard

import os
import tempfile
import unittest

import numpy as np

try:
    import torch
    import sentence_transformers  # noqa: F401 (imported by DMRL.fit)
    import tensorboard  # noqa: F401

    run_dmrl_test_funcs = True
except ImportError:
    run_dmrl_test_funcs = False

from cornac.data import ImageModality
from cornac.eval_methods import BaseMethod
from cornac.models import DMRL, Recommender


@unittest.skipUnless(run_dmrl_test_funcs, "DMRL requirements or tensorboard are not installed")
class TestDMRLSaveLoad(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        items = [f"i{j}" for j in range(30)]
        data = [
            (f"u{i}", items[j], 1.0)
            for i in range(20)
            for j in rng.choice(30, 6, replace=False)
        ]
        image = ImageModality(features=rng.rand(30, 16).astype(np.float32), ids=items)
        self.train_set = BaseMethod.from_splits(
            train_data=data,
            test_data=data[:10],
            exclude_unknowns=True,
            item_image=image,
            seed=1,
        ).train_set
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp_dir.name)  # the tensorboard writer logs under ./temp

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp_dir.cleanup()

    def test_save_and_load_with_log_metrics(self):
        model = DMRL(
            batch_size=16,
            epochs=1,
            log_metrics=True,
            bert_text_dim=0,
            image_dim=16,
            embedding_dim=8,
            num_neg=2,
            num_factors=2,
        )
        model.fit(self.train_set)

        model_file = model.save(os.path.join(self.tmp_dir.name, "saved"))
        loaded = Recommender.load(model_file)

        for p1, p2 in zip(
            model.model.state_dict().values(), loaded.model.state_dict().values()
        ):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
