# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import scipy.sparse as sp
import numpy as np

from . import FeatureModule


class GraphModule(FeatureModule):
    """Graph module
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_data = kwargs.get('data', None)
        self.__matrix = None
        self.__matrix_size = None

    @property
    def matrix(self):
        if self.__matrix is None:
            assert self.__matrix_size is not None
            self.__matrix = sp.csr_matrix((self.val, (self.map_rid, self.map_cid)),
                                          shape=(self.__matrix_size, self.__matrix_size))
        return self.__matrix

    def _build_triplet(self, id_map):
        """Build adjacency matrix in sparse triplet format using mapped ids
        """
        self.map_rid = []
        self.map_cid = []
        self.val = []
        for i, j, v in self.raw_data:
            if (i not in id_map) or (j not in id_map):
                continue
            self.map_rid.append(id_map[i])
            self.map_cid.append(id_map[j])
            self.val.append(v)

        self.map_rid = np.asarray(self.map_rid, dtype=np.int)
        self.map_cid = np.asarray(self.map_cid, dtype=np.int)
        self.val = np.asarray(self.val, dtype=np.float)

    def build(self, id_map=None):
        super().build(id_map)
        
        if id_map is not None:
            self.__matrix_size = int(max(id_map.values()) + 1)
            self._build_triplet(id_map)
        return self

    def get_train_triplet(self, train_row_ids, train_col_ids):
        """Get the training tuples
        """
        picked_idx = []
        train_row_ids = set(train_row_ids) if not isinstance(train_row_ids, set) else train_row_ids
        train_col_ids = set(train_col_ids) if not isinstance(train_col_ids, set) else train_col_ids
        for idx, (i, j) in enumerate(zip(self.map_rid, self.map_cid)):
            if (i not in train_row_ids) or (j not in train_col_ids):
                continue
            picked_idx.append(idx)

        return self.map_cid[picked_idx], \
               self.map_rid[picked_idx], \
               self.val[picked_idx]

    # TODO: add feature_fallback decorator and rename the API more meaningful
    def batch(self, batch_ids):
        """Return batch of vectors from the sparse adjacency matrix corresponding to provided batch_ids.

        Parameters
        ----------
        batch_ids: array, required
            An array contains the ids of rows to be returned from the sparse adjacency matrix.
        """

        return self.matrix[batch_ids]
