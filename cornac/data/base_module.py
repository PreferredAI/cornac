# -*- coding: utf-8 -*-

"""
@author: Quoc-Tuan Truong <tuantq.vnu@gmail.com>
"""

class BaseModule:
    """Base module

    """

    def __init__(self, id_data_map):
        self._id_to_idx = {}
        self._data = []
        for id, data in id_data_map.items():
            self._id_to_idx.setdefault(id, len(self._data))
            self._data.append(data)


    def batch(self, batch_ids):
        return self._data[batch_ids]