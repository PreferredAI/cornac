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

from . import Modality
from collections import OrderedDict


class SentimentModality(Modality):
    """Aspect module
    Parameters
    ----------
    data: List[tuple], required
        A triplet list of user, item, and sentiment information \
        which also a triplet list of aspect, opinion, and sentiment, \
        e.g., data=[('user1', 'item1', [('aspect1', 'opinion1', 'sentiment1')])].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_data = kwargs.get('data', OrderedDict())

    @property
    def sentiment(self):
        return self.__sentiment

    @sentiment.setter
    def sentiment(self, input_sentiment):
        self.__sentiment = input_sentiment

    @property
    def num_aspects(self):
        """Return the number of aspects"""
        return len(self.aspect_id_map)

    @property
    def num_opinions(self):
        """Return the number of aspects"""
        return len(self.opinion_id_map)

    @property
    def user_sentiment(self):
        return self.__user_sentiment

    @user_sentiment.setter
    def user_sentiment(self, input_user_sentiment):
        self.__user_sentiment = input_user_sentiment

    @property
    def item_sentiment(self):
        return self.__item_sentiment

    @item_sentiment.setter
    def item_sentiment(self, input_item_sentiment):
        self.__item_sentiment = input_item_sentiment

    @property
    def aspect_id_map(self):
        return self.__aspect_id_map

    @aspect_id_map.setter
    def aspect_id_map(self, input_aspect_id_map):
        self.__aspect_id_map = input_aspect_id_map

    @property
    def opinion_id_map(self):
        return self.__opinion_id_map

    @opinion_id_map.setter
    def opinion_id_map(self, input_opinion_id_map):
        self.__opinion_id_map = input_opinion_id_map

    def _build_sentiment(self, uid_map, iid_map, dok_matrix):
        self.user_sentiment = OrderedDict()
        self.item_sentiment = OrderedDict()
        aid_map = OrderedDict()
        oid_map = OrderedDict()
        sentiment = OrderedDict()
        for idx, (raw_uid, raw_iid, sentiment_tuples) in enumerate(self.raw_data):
            user_idx = uid_map.get(raw_uid, None)
            item_idx = iid_map.get(raw_iid, None)
            if user_idx is None or item_idx is None or dok_matrix[user_idx, item_idx] == 0:
                continue
            user_dict = self.user_sentiment.setdefault(user_idx, OrderedDict())
            user_dict[item_idx] = idx
            item_dict = self.item_sentiment.setdefault(item_idx, OrderedDict())
            item_dict[user_idx] = idx

            mapped_tup = []
            for tup in sentiment_tuples:
                aspect, opinion, polarity = tup[0], tup[1], float(tup[2])
                aspect_idx = aid_map.setdefault(aspect, len(aid_map))
                opinion_idx = oid_map.setdefault(opinion, len(oid_map))
                mapped_tup.append((aspect_idx, opinion_idx, polarity))
            sentiment.setdefault(idx, mapped_tup)

        self.sentiment = sentiment
        self.aspect_id_map = aid_map
        self.opinion_id_map = oid_map

    def build(self, uid_map=None, iid_map=None, dok_matrix=None, **kwargs):
        """Build the model based on provided list of ordered ids
        """
        if uid_map is not None and iid_map is not None and dok_matrix is not None:
            self._build_sentiment(uid_map, iid_map, dok_matrix)
        return self
