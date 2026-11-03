import unittest

import numpy as np

from cornac.data import Dataset
from cornac.models import BPR, WBPR


class TestBPRInitParams(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        data = [
            (str(u), str(i), 1.0)
            for u in range(20)
            for i in rng.choice(30, 5, replace=False)
        ]
        self.train_set = Dataset.from_uir(data, seed=0)
        self.init_params = {
            "U": rng.uniform(-0.1, 0.1, (self.train_set.num_users, 5)).astype(np.float32),
            "V": rng.uniform(-0.1, 0.1, (self.train_set.num_items, 5)).astype(np.float32),
            "Bi": np.zeros(self.train_set.num_items, dtype=np.float32),
        }

    def _model(self, cls, init_params):
        return cls(k=5, max_iter=3, num_threads=1, seed=1, verbose=False, init_params=init_params)

    def test_fit_leaves_given_arrays_unchanged(self):
        for cls in (BPR, WBPR):
            given = {name: value.copy() for name, value in self.init_params.items()}
            model = self._model(cls, given).fit(self.train_set)
            for name, original in self.init_params.items():
                np.testing.assert_array_equal(given[name], original)
                np.testing.assert_array_equal(model.init_params[name], original)
            self.assertFalse(np.array_equal(model.u_factors, self.init_params["U"]))
            self.assertFalse(np.array_equal(model.i_factors, self.init_params["V"]))

    def test_clone_after_fit_starts_from_the_given_values(self):
        given = {name: value.copy() for name, value in self.init_params.items()}
        model = self._model(BPR, given).fit(self.train_set)
        clone = model.clone()
        np.testing.assert_array_equal(clone.u_factors, self.init_params["U"])
        np.testing.assert_array_equal(clone.i_factors, self.init_params["V"])
        np.testing.assert_array_equal(clone.i_biases, self.init_params["Bi"])

    def test_missing_params_are_still_random(self):
        model = self._model(BPR, {}).fit(self.train_set)
        self.assertEqual(model.u_factors.shape, (self.train_set.num_users, 5))


if __name__ == "__main__":
    unittest.main()
