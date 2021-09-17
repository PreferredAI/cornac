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

import numpy as np

from cornac.data import GraphModality
from cornac.data import Reader
from collections import OrderedDict


class TestGraphModality(unittest.TestCase):
    def test_init(self):
        data = Reader().read("./tests/graph_data.txt", sep=" ")
        gmd = GraphModality(data=data)

        self.assertEqual(len(gmd.raw_data), 7)

    def test_build(self):
        data = Reader().read("./tests/graph_data.txt", sep=" ")
        gmd = GraphModality(data=data)

        global_iid_map = OrderedDict()
        for raw_iid, raw_jid, val in data:
            global_iid_map.setdefault(raw_iid, len(global_iid_map))

        gmd.build(id_map=global_iid_map)

        self.assertEqual(len(gmd.map_rid), 7)
        self.assertEqual(len(gmd.map_cid), 7)
        self.assertEqual(len(gmd.val), 7)
        self.assertEqual(gmd.matrix.shape, (7, 7))

        try:
            GraphModality().build()
        except ValueError:
            assert True

    def test_get_train_triplet(self):
        data = Reader().read("./tests/graph_data.txt", sep=" ")
        gmd = GraphModality(data=data)

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

    def test_from_feature(self):
        # build a toy feature matrix
        F = np.zeros((4, 3))
        F[0] = np.asarray([1, 0, 0])
        F[1] = np.asarray([1, 1, 0])
        F[2] = np.asarray([0, 0, 1])
        F[3] = np.asarray([1, 1, 1])

        # the expected output graph, if using cosine similarity
        s = set()
        s.update(
            [
                (0, 1, 1.0),
                (0, 3, 1.0),
                (1, 0, 1.0),
                (1, 3, 1.0),
                (2, 0, 1.0), # sim = 0
                (2, 1, 1.0), # sim = 0
                (2, 3, 1.0),
                (3, 0, 1.0), # sim ~ 0.57
                (3, 1, 1.0),
                (3, 2, 1.0), # sim ~ 0.57
            ]
        )

        # build graph modality from features
        gm = GraphModality.from_feature(features=F, k=2, similarity="cosine", verbose=False)

        self.assertTrue(isinstance(gm, GraphModality))
        self.assertTrue(not bool(gm.raw_data.difference(s)))

    def test_get_node_degree(self):
        data = Reader().read("./tests/graph_data.txt", sep=" ")
        gmd = GraphModality(data=data)

        global_iid_map = OrderedDict()
        for raw_iid, raw_jid, val in data:
            global_iid_map.setdefault(raw_iid, len(global_iid_map))

        gmd.build(id_map=global_iid_map)

        degree = gmd.get_node_degree()

        self.assertEqual(degree.get(0)[0], 4)
        self.assertEqual(degree.get(0)[1], 1)
        self.assertEqual(degree.get(1)[0], 2)
        self.assertEqual(degree.get(1)[1], 1)
        self.assertEqual(degree.get(5)[0], 0)
        self.assertEqual(degree.get(5)[1], 1)

        degree = gmd.get_node_degree([0, 1], [0, 1])

        self.assertEqual(degree.get(0)[0], 1)
        self.assertEqual(degree.get(0)[1], 0)
        self.assertEqual(degree.get(1)[0], 0)
        self.assertEqual(degree.get(1)[1], 1)


if __name__ == "__main__":
    unittest.main()
