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
        self.map_data = []
        
        
    def _build_triplet(self, ordered_ids):
        """Build adjacency matrix in sparse triplet format using maped ids
        """
        for i, j, val in self.raw_data:
            self.map_data.append([ordered_ids[i], ordered_ids[j], val])
        self.raw_data.clear()
        


    def build(self, ordered_ids):
        self._build_triplet(self,ordered_ids) 
        