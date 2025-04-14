# Copyright 2023 The Cornac Authors. All Rights Reserved.
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

import os
from collections import Counter

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from tqdm.auto import trange

from ..recommender import NextBasketRecommender


class Beacon(NextBasketRecommender):
    """Correlation-Sensitive Next-Basket Recommendation

    Parameters
    ----------
    name: string, default: 'Beacon'
        The name of the recommender model.

    emb_dim: int, optional, default: 2
        Embedding dimension

    rnn_unit: int, optional, default: 4
        Number of dimension in a rnn unit.

    alpha: float, optional, default: 0.5
        Hyperparameter to control the balance between correlative and sequential associations.

    rnn_cell_type: str, optional, default: 'LSTM'
        RNN cell type, options including ['LSTM', 'GRU', None]
        If None, BasicRNNCell will be used.

    dropout_rate: float, optional, default: 0.5
        Dropout rate of neural network dense layers

    nb_hop: int, optional, default: 1
        Number of hops for constructing correlation matrix.
        If 0, zeros matrix will be used.

    max_seq_length: int, optional, default: None
        Maximum basket sequence length.
        If None, it is the maximum number of basket in training sequences.

    n_epochs: int, optional, default: 15
        Number of training epochs

    batch_size: int, optional, default: 32
        Batch size

    lr: float, optional, default: 0.001
        Initial value of learning rate for the optimizer.

    verbose: boolean, optional, default: False
        When True, running logs are displayed.

    seed: int, optional, default: None
        Random seed

    References
    ----------
    LE, Duc Trong, Hady Wirawan LAUW, and Yuan Fang.
    Correlation-sensitive next-basket recommendation.
    International Joint Conferences on Artificial Intelligence, 2019.

    """

    def __init__(
        self,
        name="Beacon",
        emb_dim=2,
        rnn_unit=4,
        alpha=0.5,
        rnn_cell_type="LSTM",
        dropout_rate=0.5,
        nb_hop=1,
        max_seq_length=None,
        n_epochs=15,
        batch_size=32,
        lr=0.001,
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.nb_hop = nb_hop
        self.emb_dim = emb_dim
        self.rnn_unit = rnn_unit
        self.alpha = alpha
        self.rnn_cell_type = rnn_cell_type
        self.dropout_rate = dropout_rate
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.lr = lr

    def fit(self, train_set, val_set=None):
        super().fit(train_set=train_set, val_set=val_set)
        import tensorflow.compat.v1 as tf

        from .beacon_tf import BeaconModel

        tf.disable_eager_execution()

        # less verbose TF
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.logging.set_verbosity(tf.logging.ERROR)

        # max sequence length
        self.max_seq_length = (
            max([len(bids) for bids in train_set.user_basket_data.values()])
            if self.max_seq_length is None  # init max_seq_length
            else self.max_seq_length
        )
        self.correlation_matrix = self._build_correlation_matrix(
            train_set=train_set, val_set=val_set, n_items=self.total_items
        )
        self.item_probs = self._compute_item_probs(
            train_set=train_set, val_set=val_set, n_items=self.total_items
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        self.model = BeaconModel(
            sess,
            self.emb_dim,
            self.rnn_unit,
            self.alpha,
            self.max_seq_length,
            self.total_items,
            self.item_probs,
            self.correlation_matrix,
            self.rnn_cell_type,
            self.dropout_rate,
            self.seed,
            self.lr,
        )

        sess.run(tf.global_variables_initializer())  # init variable

        last_loss = np.inf
        last_val_loss = np.inf
        loop = trange(self.n_epochs, disable=not self.verbose)
        loop.set_postfix(
            loss=last_loss,
            val_loss=last_val_loss,
        )
        train_pool = []
        validation_pool = []
        for _ in loop:
            train_loss = 0.0
            trained_cnt = 0
            for batch_basket_items in self._data_iter(
                train_set, shuffle=True, current_pool=train_pool
            ):
                s, s_length, y = self._transform_data(
                    batch_basket_items, self.total_items
                )
                loss = self.model.train_batch(s, s_length, y)
                current_batch_size = len(batch_basket_items)
                trained_cnt += current_batch_size
                train_loss += loss * current_batch_size
                last_loss = train_loss / trained_cnt
                loop.set_postfix(
                    loss=last_loss,
                    val_loss=last_val_loss,
                )

            if val_set is not None:
                val_loss = 0.0
                val_cnt = 0
                for batch_basket_items in self._data_iter(
                    val_set, shuffle=False, current_pool=validation_pool
                ):
                    s, s_length, y = self._transform_data(
                        batch_basket_items, self.total_items
                    )
                    loss = self.model.validate_batch(s, s_length, y)
                    current_batch_size = len(batch_basket_items)
                    val_cnt += current_batch_size
                    val_loss += loss * current_batch_size
                    last_val_loss = val_loss / val_cnt
                    loop.set_postfix(
                        loss=last_loss,
                        val_loss=last_val_loss,
                    )

        return self

    def _data_iter(self, data_set, shuffle=False, current_pool=[]):
        """This iterator ensure each batch has same size, the remaining data will be preceded in the next epoch"""
        for _, _, batch_basket_items in data_set.ubi_iter(
            batch_size=self.batch_size, shuffle=shuffle
        ):
            current_pool += batch_basket_items
            if len(current_pool) >= self.batch_size:
                yield current_pool[: self.batch_size]
                del current_pool[self.batch_size :]

    def _transform_data(self, batch_basket_items, n_items):
        assert len(batch_basket_items) == self.batch_size
        s = [basket_items[:-1] for basket_items in batch_basket_items]
        s_length = [len(b) for b in s]
        y = np.zeros((self.batch_size, n_items), dtype="int32")
        for inc, basket_items in enumerate(batch_basket_items):
            y[inc, basket_items[-1]] = 1
        return s, s_length, y

    def _build_correlation_matrix(self, train_set, val_set, n_items):
        if self.nb_hop == 0:
            return csr_matrix((n_items, n_items), dtype="float32")

        pairs_cnt = Counter()
        for _, _, [basket_items] in train_set.ubi_iter(1, shuffle=False):
            for items in basket_items:
                current_items = np.unique(items)
                for i in range(len(current_items) - 1):
                    for j in range(i + 1, len(current_items)):
                        pairs_cnt[(current_items[i], current_items[j])] += 1
        if val_set is not None:
            for _, _, [basket_items] in val_set.ubi_iter(1, shuffle=False):
                for items in basket_items:
                    current_items = np.unique(items)
                    for i in range(len(current_items) - 1):
                        for j in range(i + 1, len(current_items)):
                            pairs_cnt[(current_items[i], current_items[j])] += 1
        data, row, col = [], [], []
        for pair, cnt in pairs_cnt.most_common():
            data.append(cnt)
            row.append(pair[0])
            col.append(pair[1])
        correlation_matrix = csc_matrix(
            (data, (row, col)), shape=(n_items, n_items), dtype="float32"
        )
        correlation_matrix = self._normalize(correlation_matrix)

        w_mul = correlation_matrix
        coeff = 1.0
        for _ in range(1, self.nb_hop):
            coeff *= 0.85
            w_mul *= correlation_matrix
            w_mul = self._remove_diag(w_mul)
            w_adj_matrix = self._normalize(w_mul)
            correlation_matrix += coeff * w_adj_matrix

        return correlation_matrix

    def _remove_diag(self, adj_matrix):
        new_adj_matrix = csr_matrix(adj_matrix)
        new_adj_matrix.setdiag(0.0)
        new_adj_matrix.eliminate_zeros()
        return new_adj_matrix

    def _normalize(self, adj_matrix: csr_matrix):
        """Symmetrically normalize adjacency matrix."""
        row_sum = adj_matrix.sum(1).toarray().squeeze()
        d_inv_sqrt = np.power(
            row_sum,
            -0.5,
            out=np.zeros_like(row_sum, dtype="float32"),
            where=row_sum != 0,
        )
        d_mat_inv_sqrt = diags(d_inv_sqrt)

        normalized_matrix = (
            adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        )

        return normalized_matrix.tocsr()

    def _compute_item_probs(self, train_set, val_set, n_items):
        item_freq = Counter(train_set.uir_tuple[1])
        if val_set is not None:
            item_freq += Counter(val_set.uir_tuple[1])
        item_probs = np.zeros(n_items, dtype="float32")
        total_cnt = len(train_set.uir_tuple[1]) + len(val_set.uir_tuple[1])
        for iid, cnt in item_freq.items():
            item_probs[iid] = cnt / total_cnt
        return item_probs

    def score(self, user_idx, history_baskets, **kwargs):
        s = [history_baskets]
        s_length = [len(history_baskets)]
        return self.model.predict(s, s_length)
