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
        self.map_rid = []
        self.map_cid = []
        self.val = []

    def _build_triplet(self, global_id_map):
        """Build adjacency matrix in sparse triplet format using mapped ids
        """

        for i, j, v in self.raw_data:
            self.map_rid.append(global_id_map[i])
            self.map_cid.append(global_id_map[j])
            self.val.append(v)
            
            #self.map_data = (np.asarray(self.map_rid, dtype=np.int),
            #                 np.asarray(self.map_cid, dtype=np.int),
            #                 np.asarray(self.val, dtype=np.float))
            
        self.map_rid = np.asarray(self.map_rid, dtype=np.int)
        self.map_cid = np.asarray(self.map_cid, dtype=np.int)
        self.val = np.asarray(self.val, dtype=np.float)


    def _build_sparse_matrix(self, triplet):
        """Build sparse adjacency matrix
        """
        (rid, cid, val) = triplet
        n_rows = int(max(rid) + 1)
        n_cols = int(max(cid) + 1)

        self.matrix = sp.csr_matrix((val, (rid, cid)), shape=(n_rows, n_cols))

    def get_train_triplet(self, train_row_ids, train_col_ids):
        """Get the training tuples
        """
        rid = []
        cid = []
        val = []
        train_triplet = []
        # this makes operations much more efficient
        train_row_ids = np.asanyarray(list(train_row_ids))
        train_col_ids = np.asanyarray(list(train_col_ids))
        for i, j, v in zip(self.map_rid, self.map_cid, self.val):
            if (i not in train_row_ids) or (j not in train_col_ids):
                continue
            rid.append(i)
            cid.append(j)
            val.append(v)
            
        train_triplet = (np.asarray(rid, dtype=np.int),
        np.asarray(cid, dtype=np.int),
        np.asarray(val, dtype=np.float))

        return train_triplet

    # TODO: id_map can be None to support GraphModule as an independent component
    def build(self, id_map=None):
        self._build_triplet(id_map)
        self._build_sparse_matrix((self.map_rid, self.map_cid, self.val))

    # TODO: add feature_fallback decorator and rename the API more meaningful
    def batch(self, batch_ids):
        """Return batch of vectors from the sparse adjacency matrix corresponding to provided batch_ids.

        Parameters
        ----------
        batch_ids: array, required
            An array contains the ids of rows to be returned from the sparse adjacency matrix.
        """

        return self.matrix[batch_ids]
