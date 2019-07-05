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

import numpy as np


class Dataset:

    def __init__(self, data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        self.index = None
        pass

    @property
    def data(self):
        return self._data

    def index_trans(self):
        self._data = np.unique(self._data, axis=0)
        valid_users = list(np.unique(self._data[:, 0]))
        valid_items = list(np.unique(self._data[:, 1]))
        mylist = []
        for row in self._data:
            mylist.append([valid_users.index(row[0]), valid_items.index(row[1]), row[2]])
        mat = np.array(mylist)
        return mat, valid_users, valid_items

    # in this version we do not shuffle the original data (only the ids)
    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            print('Shafling the data')
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            # np.random.shuffle(idx)  # shuffle indexe
            self.index = idx
            # self._data = self.data[idx]  # get list of `num` random samples

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            # data_rest_part = self.data[self.index][start:self._num_examples]
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            # np.random.shuffle(idx)  # shuffle indexes
            # self._data = self.data[idx0]  # get list of `num` random samples
            idex_rest = self.index[start:self._num_examples]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self._index_in_epoch
            # data_new_part =  self._data[idx0][start:end]
            # data_new_part =  self._data[start:end]
            index_new = idx[start:end]
            self.index = idx
            # return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((idex_rest, index_new))
            return self._data[np.concatenate((idex_rest, index_new))], np.concatenate((idex_rest, index_new))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            # alose return the ids
            return self._data[self.index[start:end]], self.index[start:end]
            # return self._data[start:end], self.index[start:end]
