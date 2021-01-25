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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class NARRE(Recommender):
    """Neural Attentional Rating Regression with Review-level Explanations

    Parameters
    ----------
    name: string, default: 'NARRE'
        The name of the recommender model.

    embedding_size: int, default: 100
        Word embedding size

    id_embedding_size: int, default: 32
        User/item review id embedding size

    n_factors: int, default: 32
        The dimension of the user/item's latent factors.

    attention_size: int, default: 16
        Attention size

    kernel_sizes: list, default: [3]
        List of kernel sizes of conv2d

    n_filters: int, default: 64
        Number of filters

    dropout_rate: float, default: 0.5
        Dropout rate of neural network dense layers

    max_text_length: int, default: 50
        Maximum number of tokens in a review instance

    max_num_review: int, default: None
        Maximum number of reviews that you want to feed into training. By default, the model will be trained with all reviews.

    batch_size: int, default: 64
        Batch size

    max_iter: int, default: 10
        Max number of training epochs

    optimizer: string, optional, default: 'adam'
        Optimizer for training is either 'adam' or 'rmsprop'.

    learning_rate: float, optional, default: 0.001
        Initial value of learning rate for the optimizer.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, pretrained_word_embeddings could be initialized here, e.g., init_params={'pretrained_word_embeddings': pretrained_word_embeddings}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Chen, C., Zhang, M., Liu, Y., & Ma, S. (2018, April). Neural attentional rating regression with review-level explanations. In Proceedings of the 2018 World Wide Web Conference (pp. 1583-1592).
    """

    def __init__(
        self,
        name="NARRE",
        embedding_size=100,
        id_embedding_size=32,
        n_factors=32,
        attention_size=16,
        kernel_sizes=[3],
        n_filters=64,
        dropout_rate=0.5,
        max_text_length=50,
        max_num_review=None,
        batch_size=64,
        max_iter=10,
        optimizer='adam',
        learning_rate=0.001,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.seed = seed
        self.embedding_size = embedding_size
        self.id_embedding_size = id_embedding_size
        self.n_factors = n_factors
        self.attention_size = attention_size
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.max_text_length = max_text_length
        self.max_num_review = max_num_review
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        # Init params if provided
        self.init_params = {} if init_params is None else init_params

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

        if self.trainable:
            if not hasattr(self, "model"):
                from .narre import Model
                self.model = Model(
                    self.train_set.num_users,
                    self.train_set.num_items,
                    self.train_set.review_text.vocab,
                    self.train_set.global_mean,
                    n_factors=self.n_factors,
                    embedding_size=self.embedding_size,
                    id_embedding_size=self.id_embedding_size,
                    attention_size=self.attention_size,
                    kernel_sizes=self.kernel_sizes,
                    n_filters=self.n_filters,
                    dropout_rate=self.dropout_rate,
                    max_text_length=self.max_text_length,
                    pretrained_word_embeddings=self.init_params.get('pretrained_word_embeddings'),
                    verbose=self.verbose,
                    seed=self.seed,
                )
            self._fit_narre()

        return self

    def _fit_narre(self):
        import tensorflow as tf
        from tensorflow import keras
        from .narre import get_data
        loss = keras.losses.MeanSquaredError()
        if self.optimizer == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_loss = keras.metrics.Mean(name="loss")
        loop = trange(self.max_iter, disable=not self.verbose)
        for _ in loop:
            train_loss.reset_states()
            for i, (batch_users, batch_items, batch_ratings) in enumerate(self.train_set.uir_iter(self.batch_size, shuffle=True)):
                user_reviews, user_iid_reviews, user_num_reviews = get_data(batch_users, self.train_set, self.max_text_length, by='user', max_num_review=self.max_num_review)
                item_reviews, item_uid_reviews, item_num_reviews = get_data(batch_items, self.train_set, self.max_text_length, by='item', max_num_review=self.max_num_review)
                with tf.GradientTape() as tape:
                    predictions = self.model.graph(
                        [batch_users, batch_items, user_reviews, user_iid_reviews, user_num_reviews, item_reviews, item_uid_reviews, item_num_reviews],
                        training=True,
                    )
                    _loss = loss(batch_ratings, predictions)
                gradients = tape.gradient(_loss, self.model.graph.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.graph.trainable_variables))
                train_loss(_loss)
                if i % 10 == 0:
                    loop.set_postfix(loss=train_loss.result().numpy())
        loop.close()

        # save weights for predictions
        self.X, self.Y, self.W1, self.user_embedding, self.item_embedding, self.bu, self.bi, self.mu = self.model.get_weights(self.train_set, self.batch_size, max_num_review=self.max_num_review)
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
        model = self.model
        del self.model

        model_file = Recommender.save(self, save_dir)

        self.model = model
        self.model.save(model_file.replace(".pkl", ".cpt"))

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
        from tensorflow import keras
        model = Recommender.load(model_path, trainable)
        model.model = keras.models.load_model(model.load_from.replace(".pkl", ".cpt"))

        return model

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
            h0 = (self.user_embedding[user_idx] + self.X[user_idx]) * (self.item_embedding + self.Y)
            known_item_scores = h0.dot(self.W1) + self.bu[user_idx] + self.bi + self.mu
            return known_item_scores.ravel()
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            h0 = (self.user_embedding[user_idx] + self.X[user_idx]) * (self.item_embedding[item_idx] + self.Y[item_idx])
            known_item_score = h0.dot(self.W1) + self.bu[user_idx] + self.bi[item_idx] + self.mu
            return known_item_score
