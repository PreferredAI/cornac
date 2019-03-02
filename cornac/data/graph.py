# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

from . import Module

class GraphModule(Module):
    """Graph module

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_data = kwargs.get('data', None)
        self.map_data = None
        
        
    def _build_graph(self, ordered_ids):
        """Build adjacency matrix in sparse triplet format using maped ids
        """

        self.data_feature = np.zeros((len(ordered_ids), self.feature_dim))
        for map_id, raw_id in enumerate(ordered_ids):
            self.data_feature[map_id] = self._id_feature[raw_id]
        if self._normalized:
            self.data_feature = self.data_feature - np.min(self.data_feature)
            self.data_feature = self.data_feature / (np.max(self.data_feature) + 1e-10)

        self._id_feature.clear()


    def build(self, ordered_ids):
        pass 
        