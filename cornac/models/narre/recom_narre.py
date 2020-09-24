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

import os
import numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException
from ...utils import get_rng, estimate_batches
from ...utils.init_utils import xavier_uniform


class NARRE(Recommender):
    """Neural Attentional Rating Regression with Review-level Explanations

    Parameters
    ----------
    name: string, default: 'NARRE'
        The name of the recommender model.

    review_num_u: int, default: 32
        Number of review of user
    review_num_i: int, default: 32
        Number of review of item
    review_len_u: int, default: 500
        Max length each user review text
    review_len_i: int, default: 500
        Max length each item review text
    n_latent: int, default: 32
        Latent dimension size
    attention_size: int, default: 32
        Attention dimension size
    word_embedding_size: int, default: 300
        Word embedding size
    id_embedding_size: int, default: 32
        Id embedding size
    filter_sizes: list, default: [3]
        List of filter sizes
    n_filters: int, default: 100
        Number of filters
    dropout: float, default: 0.8
        Dropout ratio
    dropout_keep_prob: float, default:0.5
        Dropout ratio in ncf
    allow_soft_placement: bool, default: True
    log_device_placement: bool, default: False
    l2_reg_lambda: float, default: 0.001
        L2 regularization factor
    batch_size: int, default: 100
        Batch size
    max_iter: int, default: 50,
        Max number of training epochs

    References
    ----------
    * Chen, C., Zhang, M., Liu, Y., & Ma, S. (2018, April). Neural attentional rating regression with review-level explanations. In Proceedings of the 2018 World Wide Web Conference (pp. 1583-1592).
    """

    def __init__(
        self,
        name="NARRE",
        review_num_u=32,
        review_num_i=32,
        review_len_u=500,
        review_len_i=500,
        n_latent=32,
        attention_size=32,
        word_embedding_size=300,
        id_embedding_size=32,
        filter_sizes=[3],
        n_filters=100,
        dropout=0.8,
        dropout_keep_prob=0.5,
        allow_soft_placement=True,
        log_device_placement=False,
        l2_reg_lambda=0.001,
        batch_size=100,
        max_iter=50,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.seed = seed
        self.rng = get_rng(seed)
        self.review_num_u = review_num_u
        self.review_num_i = review_num_i
        self.review_len_u = review_len_u
        self.review_len_i = review_len_i
        self.n_latent = n_latent
        self.attention_size = attention_size
        self.word_embedding_size = word_embedding_size
        self.id_embedding_size = id_embedding_size
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.max_iter = max_iter
        self.dropout = dropout
        self.batch_size = batch_size
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.dropout_keep_prob = dropout_keep_prob
        # Init params if provided
        self.init_params = {} if init_params is None else init_params

    def _init(self):
        self.n_users, self.n_items = self.train_set.num_users, self.train_set.num_items
        self.pretrained_word_embeddings = self.init_params.get('pretrained_word_embeddings')

    def _build_graph(self):
        from .narre import Model
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        # less verbose TF
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            self.vocab = self.train_set.review_text.vocab
            self.user_review = self.train_set.review_text.user_review
            self.item_review = self.train_set.review_text.item_review

            self.model = Model(
                review_num_u=self.review_num_u,
                review_num_i=self.review_num_i,
                review_len_u=self.review_len_u,
                review_len_i=self.review_len_i,
                n_users=self.n_users,
                n_items=self.n_items,
                user_vocabulary=self.vocab,
                item_vocabulary=self.vocab,
                n_latent=self.n_latent,
                id_embedding_size=self.id_embedding_size,
                attention_size=self.attention_size,
                word_embedding_size=self.word_embedding_size,
                filter_sizes=self.filter_sizes,
                n_filters=self.n_filters,
                l2_reg_lambda=self.l2_reg_lambda,
                pretrained_word_embeddings=self.pretrained_word_embeddings
            )
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self._sess_init()

    def _sess_init(self):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        config = tf.ConfigProto(
            allow_soft_placement=self.allow_soft_placement,
            log_device_placement=self.log_device_placement,
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.initializer)

    def _get_data(self, batch_users, batch_items):
        batch_user_reviews, batch_item_reviews = [], []
        review_uids, review_iids = [], []
        for iid in batch_items:
            user_ids, review_ids = np.ones(self.review_num_i) * self.n_users, []
            for idx, (user_idx, review_idx) in enumerate(self.train_set.review_text.item_review[iid].items()):
                if idx + 1 > self.review_num_i:
                    break
                user_ids[idx] = user_idx
                review_ids.append(review_idx)
            review_uids.append(user_ids)
            mapped_reviews = self.train_set.review_text.batch_seq(review_ids, max_length=self.review_len_i)
            item_reviews = np.zeros((self.review_num_i, self.review_len_i))
            item_reviews[:len(review_ids)] = mapped_reviews
            batch_user_reviews.append(item_reviews)
        for uid in batch_users:
            item_ids, review_ids = np.ones(self.review_num_u) * self.n_items, []
            for idx, (item_idx, review_idx) in enumerate(self.train_set.review_text.user_review[uid].items()):
                if idx + 1 > self.review_num_u:
                    break
                item_ids[idx] = item_idx
                review_ids.append(review_idx)
            review_iids.append(item_ids)
            mapped_reviews = self.train_set.review_text.batch_seq(review_ids, max_length=self.review_len_u)
            user_reviews = np.zeros((self.review_num_u, self.review_len_u))
            user_reviews[:len(review_ids)] = mapped_reviews
            batch_item_reviews.append(user_reviews)
        batch_item_reviews = np.array(batch_item_reviews, dtype=np.int32)
        batch_user_reviews = np.array(batch_user_reviews, dtype=np.int32)
        review_uids = np.array(review_uids, dtype=np.int32)
        review_iids = np.array(review_iids, dtype=np.int32)
        return batch_user_reviews, batch_item_reviews, review_uids, review_iids

    def _step_update(self, batch_users, batch_items, batch_ratings):
        batch_user_reviews, batch_item_reviews, review_uids, review_iids = self._get_data(batch_users, batch_items)
        _, _loss = self.sess.run(
            [self.model.opt, self.model.loss],
            feed_dict={
                self.model.input_u: batch_item_reviews,
                self.model.input_i: batch_user_reviews,
                self.model.input_uid: batch_users.reshape(-1, 1),
                self.model.input_iid: batch_items.reshape(-1, 1),
                self.model.input_y: batch_ratings.reshape(-1, 1),
                self.model.input_reuid: review_iids,
                self.model.input_reiid: review_uids,
                self.model.drop0: self.dropout,
                self.model.dropout_keep_prob: self.dropout_keep_prob,
            },
        )
        return _loss

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        self._init()

        if self.trainable:
            if not hasattr(self, "graph"):
                self._build_graph()
            self._fit_narre()

        return self

    def _fit_narre(self):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()

        loop = trange(self.max_iter, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            for i, (batch_users, batch_items, batch_ratings) in enumerate(
                self.train_set.uir_iter(self.batch_size, shuffle=True)
            ):
                _loss = self._step_update(batch_users, batch_items, batch_ratings)
                sum_loss += _loss * len(batch_ratings)
                count += len(batch_ratings)
                if i % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))

        loop.close()

        if self.verbose:
            print("Learning completed!")

    def save(self, save_dir=None):
        """Save a recommender model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

        """
        if save_dir is None:
            return

        model_file = Recommender.save(self, save_dir)
        # save TF weights
        self.saver.save(self.sess, model_file.replace(".pkl", ".cpt"))

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem.

        Parameters
        ----------
        model_path: str, required
            Path to a file or directory where the model is stored. If a directory is
            provided, the latest model will be loaded.

        trainable: boolean, optional, default: False
            Set it to True if you would like to finetune the model. By default, 
            the model parameters are assumed to be fixed after being loaded.
        
        Returns
        -------
        self : object
        """
        model = Recommender.load(model_path, trainable)
        if hasattr(model, "pretrained"):
            model.pretrained = False

        model._build_graph()
        model.saver.restore(model.sess, model.load_from.replace(".pkl", ".cpt"))

        return model

    def _idx_iter(self, idx_range, batch_size):
        indices = np.arange(idx_range)
        num_batches = estimate_batches(len(indices), batch_size)
        for b in range(num_batches):
            start_offset = batch_size * b
            end_offset = batch_size * b + batch_size
            end_offset = min(end_offset, len(indices))
            batch_ids = indices[start_offset:end_offset]
            yield batch_ids

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            batch_users = np.ones(self.train_set.num_items) * user_idx
            batch_items = np.arange(self.train_set.num_items)
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            batch_users = np.array([user_idx])
            batch_items = np.array([item_idx])
        known_item_scores = np.ones(len(batch_users))
        for batch_ids in self._idx_iter(len(batch_users), self.batch_size):
            _batch_users = batch_users[batch_ids]
            _batch_items = batch_items[batch_ids]
            batch_user_reviews, batch_item_reviews, review_uids, review_iids = self._get_data(_batch_users, _batch_items)
            _known_item_scores = self.sess.run(
                self.model.predictions,
                feed_dict={
                    self.model.input_u: batch_user_reviews,
                    self.model.input_i: batch_item_reviews,
                    self.model.input_uid: _batch_users.reshape(-1, 1),
                    self.model.input_iid: _batch_items.reshape(-1, 1),
                    self.model.input_reuid: review_uids,
                    self.model.input_reiid: review_iids,
                    self.model.dropout_keep_prob: 1,
                }
            )
            known_item_scores[batch_ids] = _known_item_scores.ravel()
        return known_item_scores.ravel()