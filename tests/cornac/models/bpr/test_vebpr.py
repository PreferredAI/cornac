
import unittest
import numpy as np
import scipy.sparse as sp
from cornac.models import VEBPR
from cornac.data import Dataset


class TestVEBPR(unittest.TestCase):
    def test_vebpr(self):
        train_set = Dataset(
            num_users=2, num_items=2,
            uid_map={"u1": 0, "u2": 1}, iid_map={"i1": 0, "i2": 1},
            uir_tuple=([0, 1], [0, 1], [1.0, 1.0])
        )
        v_matrix = sp.csr_matrix(([1.0], ([0], [1])), shape=(2, 2))

        model = VEBPR(k=2, max_iter=1, view_matrix=v_matrix)
        model.fit(train_set)

        self.assertEqual(len(model.score(0)), 2)


if __name__ == '__main__':
    unittest.main()