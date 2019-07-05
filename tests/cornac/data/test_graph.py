# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import unittest

from cornac.data import GraphModule
from cornac.data import Reader
from collections import OrderedDict


class TestGraphModule(unittest.TestCase):

    def test_init(self):
        data = Reader().read('./tests/graph_data.txt', sep=' ')
        gmd = GraphModule(data=data)

        self.assertEqual(len(gmd.raw_data), 7)

    def test_build(self):
        data = Reader().read('./tests/graph_data.txt', sep=' ')
        gmd = GraphModule(data=data)

        global_iid_map = OrderedDict()
        for raw_iid, raw_jid, val in data:
            global_iid_map.setdefault(raw_iid, len(global_iid_map))

        gmd.build(id_map=global_iid_map)

        self.assertEqual(len(gmd.map_rid), 7)
        self.assertEqual(len(gmd.map_cid), 7)
        self.assertEqual(len(gmd.val), 7)
        self.assertEqual(gmd.matrix.shape, (7, 7))

        try:
            GraphModule().build()
        except ValueError:
            assert True

    def test_get_train_triplet(self):
        data = Reader().read('./tests/graph_data.txt', sep=' ')
        gmd = GraphModule(data=data)

        global_iid_map = OrderedDict()
        for raw_iid, raw_jid, val in data:
            global_iid_map.setdefault(raw_iid, len(global_iid_map))

        gmd.build(id_map=global_iid_map)
        rid, cid, val = gmd.get_train_triplet([0, 1, 2], [0, 1, 2])
        self.assertEqual(len(rid), 3)
        self.assertEqual(len(cid), 3)
        self.assertEqual(len(val), 3)

        rid, cid, val = gmd.get_train_triplet([0, 2], [0, 1])
        self.assertEqual(len(rid), 1)
        self.assertEqual(len(cid), 1)
        self.assertEqual(len(val), 1)


if __name__ == '__main__':
    unittest.main()
