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
from cornac.data import SentimentModality
from cornac.data import Reader
from collections import OrderedDict
from scipy.sparse import dok_matrix


class TestSentimentModality(unittest.TestCase):

    def test_init(self):
        data = Reader().read('./tests/sentiment_data.txt', fmt='UITup', sep=',', tup_sep=':')
        md = SentimentModality(data=data)

        self.assertEqual(len(md.raw_data), 4)

    def test_build(self):
        data = Reader().read('./tests/sentiment_data.txt', fmt='UITup', sep=',', tup_sep=':')
        md = SentimentModality(data=data)

        uid_map = OrderedDict()
        iid_map = OrderedDict()
        for raw_uid, raw_iid, _ in data:
            uid_map.setdefault(raw_uid, len(uid_map))
            iid_map.setdefault(raw_iid, len(iid_map))

        matrix = dok_matrix((len(uid_map), len(iid_map)), dtype=np.float32)

        for raw_uid, raw_iid, _ in data:
            user_idx = uid_map.get(raw_uid)
            item_idx = iid_map.get(raw_iid)
            matrix[user_idx, item_idx] = 1

        md.build(uid_map=uid_map, iid_map=iid_map, dok_matrix=matrix)

        self.assertEqual(md.num_aspects, 3)
        self.assertEqual(md.num_opinions, 2)
        self.assertEqual(len(md.sentiment), 4)
        self.assertEqual(len(md.user_sentiment), 3)
        self.assertEqual(len(md.item_sentiment), 3)
        self.assertEqual(len(md.aspect_id_map), 3)
        self.assertEqual(len(md.opinion_id_map), 2)

        try:
            SentimentModality().build()
        except ValueError:
            assert True

if __name__ == '__main__':
    unittest.main()
