import unittest

import numpy as np

from cornac.data import Dataset
from cornac.models import MF, NMF, PMF, BaselineOnly

# Each model, its constructor arguments, and the attribute holding each init param.
MODELS = [
    (MF, {"k": 5, "use_bias": True, "num_threads": 1}, {"U": "u_factors", "V": "i_factors", "Bu": "u_biases", "Bi": "i_biases"}),
    (NMF, {"k": 5, "use_bias": True, "num_threads": 1}, {"U": "u_factors", "V": "i_factors", "Bu": "u_biases", "Bi": "i_biases"}),
    (BaselineOnly, {"num_threads": 1}, {"Bu": "u_biases", "Bi": "i_biases"}),
    (PMF, {"k": 5}, {"U": "U", "V": "V"}),
]


class TestInitParams(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        data = [
            (str(u), str(i), float(rng.randint(1, 6)))
            for u in range(20)
            for i in rng.choice(30, 5, replace=False)
        ]
        self.train_set = Dataset.from_uir(data, seed=0)
        num_users, num_items = self.train_set.num_users, self.train_set.num_items
        self.init_params = {
            "U": rng.uniform(0.1, 0.2, (num_users, 5)).astype(np.float32),
            "V": rng.uniform(0.1, 0.2, (num_items, 5)).astype(np.float32),
            "Bu": rng.uniform(-0.1, 0.1, num_users).astype(np.float32),
            "Bi": rng.uniform(-0.1, 0.1, num_items).astype(np.float32),
        }

    def _fit(self, cls, kwargs, names):
        # PMF only accepts double precision.
        dtype = np.float64 if cls is PMF else np.float32
        self.init_params = {name: value.astype(dtype) for name, value in self.init_params.items()}
        given = {name: self.init_params[name].copy() for name in names}
        model = cls(max_iter=3, seed=1, verbose=False, init_params=given, **kwargs)
        return model.fit(self.train_set), given

    def test_fit_leaves_given_arrays_unchanged(self):
        for cls, kwargs, names in MODELS:
            with self.subTest(model=cls.__name__):
                model, given = self._fit(cls, kwargs, names)
                for name, attribute in names.items():
                    np.testing.assert_array_equal(given[name], self.init_params[name])
                    np.testing.assert_array_equal(model.init_params[name], self.init_params[name])
                    # The model itself must still have been trained.
                    self.assertFalse(np.array_equal(getattr(model, attribute), self.init_params[name]))

    def test_clone_after_fit_starts_from_the_given_values(self):
        for cls, kwargs, names in MODELS:
            with self.subTest(model=cls.__name__):
                model, _ = self._fit(cls, kwargs, names)
                clone = model.clone()
                for name, attribute in names.items():
                    np.testing.assert_array_equal(getattr(clone, attribute), self.init_params[name])

    def test_refitting_a_clone_matches_the_first_fit(self):
        for cls, kwargs, names in MODELS:
            with self.subTest(model=cls.__name__):
                model, _ = self._fit(cls, kwargs, names)
                clone = model.clone().fit(self.train_set)
                for attribute in names.values():
                    np.testing.assert_allclose(getattr(clone, attribute), getattr(model, attribute))

    def test_missing_params_are_still_initialized(self):
        for cls, kwargs, names in MODELS:
            with self.subTest(model=cls.__name__):
                model = cls(max_iter=1, seed=1, verbose=False, **kwargs).fit(self.train_set)
                for attribute in names.values():
                    self.assertIsNotNone(getattr(model, attribute))


if __name__ == "__main__":
    unittest.main()
