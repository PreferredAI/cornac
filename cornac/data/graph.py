# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

from . import FeatureModule
import scipy.sparse as sp
import numpy as np


class GraphModule(FeatureModule):
    """Graph module
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_data = kwargs.get('data', None)
        self.matrix = None
        self.map_data = []

    def _build_triplet(self, global_id_map):
        """Build adjacency matrix in sparse triplet format using mapped ids
        """

        for i, j, val in self.raw_data:
            self.map_data.append([global_id_map[i], global_id_map[j], val])
        self.map_data = np.asanyarray(self.map_data)
        self.raw_data = None

    def _build_sparse_matrix(self, triplet):
        """Build sparse adjacency matrix
        """

        n_rows = max(triplet[:, 0]) + 1
        n_cols = max(triplet[:, 1]) + 1
        self.matrix = sp.csc_matrix((triplet[:, 2], (triplet[:, 0], triplet[:, 1])), shape=(n_rows, n_cols))

    def get_train_triplet(self, train_row_ids, train_col_ids):
        """Get the training tuples
        """
        train_triplet = []
        # this makes operations much more efficient
        train_row_ids = np.asanyarray(list(train_row_ids))
        train_col_ids = np.asanyarray(list(train_col_ids))
        for i, j, val in self.map_data:
            if (i not in train_row_ids) or (j not in train_col_ids):
                continue
            train_triplet.append([i, j, val])

        return np.asarray(train_triplet)

    def build(self, global_id_map):
        self._build_triplet(global_id_map)
        self._build_sparse_matrix(self.map_data)

    def batch(self, batch_ids):
        """Return batch of vectors from the sparse adjacency matrix corresponding to provided batch_ids.

        Parameters
        ----------
        batch_ids: array, required
            An array contains the ids of rows to be returned from the sparse adjacency matrix.
        """

        return self.matrix[batch_ids]
