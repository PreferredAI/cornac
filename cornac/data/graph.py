# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

from . import Module
import scipy.sparse as sp

class GraphModule(Module):
    """Graph module

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_data = kwargs.get('data', None)
        self.matrix = None
        self.map_data = []
        
        
    def _build_triplet(self, ordered_ids):
        """Build adjacency matrix in sparse triplet format using maped ids
        """
        for i, j, val in self.raw_data:
            self.map_data.append([ordered_ids[i], ordered_ids[j], val])
        self.raw_data.clear()
       

    def _build_sparse_matrix(self, triplet):
        n_rows = max(triplet[:,0]) + 1
        n_cols = max(triplet[:,1]) + 1
        self.matrix = sp.csc_matrix((triplet[:,2], (triplet[:,0], triplet[:,1])), shape=(len(n_rows), len(n_cols)))

        

    def build(self, ordered_ids):
        self._build_triplet(self,ordered_ids) 
        self._build__build_sparse_matrix(map_data)
        