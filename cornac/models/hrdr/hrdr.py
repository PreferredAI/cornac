import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from ...utils import get_rng
from ...utils.init_utils import uniform
from ..narre.narre import TextProcessor, AddGlobalBias


def get_data(batch_ids, train_set, max_text_length, by="user", max_num_review=32):
    batch_reviews, batch_num_reviews = [], []
    review_group = (
        train_set.review_text.user_review
        if by == "user"
        else train_set.review_text.item_review
    )
    for idx in batch_ids:
        review_ids = []
        for inc, (jdx, review_idx) in enumerate(review_group[idx].items()):
            if max_num_review is not None and inc == max_num_review:
                break
            review_ids.append(review_idx)
        reviews = train_set.review_text.batch_seq(
            review_ids, max_length=max_text_length
        )
        batch_reviews.append(reviews)
        batch_num_reviews.append(len(reviews))
    batch_reviews = pad_sequences(batch_reviews, maxlen=max_num_review, padding="post")
    batch_num_reviews = np.array(batch_num_reviews).astype(np.int32)
    batch_ratings = (
        np.zeros((len(batch_ids), train_set.num_items), dtype=np.float32)
        if by == "user"
        else np.zeros((len(batch_ids), train_set.num_users), dtype=np.float32)
    )
    rating_group = train_set.user_data if by == "user" else train_set.item_data
    for batch_inc, idx in enumerate(batch_ids):
        jds, ratings = rating_group[idx]
        for jdx, rating in zip(jds, ratings):
            batch_ratings[batch_inc, jdx] = rating
    return batch_reviews, batch_num_reviews, batch_ratings

class Model(keras.Model):
    def __init__(self, n_users, n_items, n_vocab, global_mean, embedding_matrix,
                 n_factors=32, embedding_size=100, id_embedding_size=32,
                 attention_size=16, kernel_sizes=[3], n_filters=64,
                 n_user_mlp_factors=128, n_item_mlp_factors=128,
                 dropout_rate=0.5, max_text_length=50):
        super().__init__()
        self.l_user_review_embedding = layers.Embedding(n_vocab, embedding_size, embeddings_initializer=embedding_matrix, mask_zero=True, name="user_review_embedding")
        self.l_item_review_embedding = layers.Embedding(n_vocab, embedding_size, embeddings_initializer=embedding_matrix, mask_zero=True, name="item_review_embedding")
        self.l_user_embedding = layers.Embedding(n_users, id_embedding_size, embeddings_initializer="uniform", name="user_embedding")
        self.l_item_embedding = layers.Embedding(n_items, id_embedding_size, embeddings_initializer="uniform", name="item_embedding")
        self.user_bias = layers.Embedding(n_users, 1, embeddings_initializer=tf.initializers.Constant(0.1), name="user_bias")
        self.item_bias = layers.Embedding(n_items, 1, embeddings_initializer=tf.initializers.Constant(0.1), name="item_bias")
        self.user_text_processor = TextProcessor(max_text_length, filters=n_filters, kernel_sizes=kernel_sizes, dropout_rate=dropout_rate, name='user_text_processor')
        self.item_text_processor = TextProcessor(max_text_length, filters=n_filters, kernel_sizes=kernel_sizes, dropout_rate=dropout_rate, name='item_text_processor')

        self.l_user_mlp = keras.models.Sequential([
            layers.Dense(n_user_mlp_factors, input_dim=n_items, activation="relu"),
            layers.Dense(n_user_mlp_factors // 2, activation="relu"),
            layers.Dense(n_filters, activation="relu"),
            layers.BatchNormalization(),
        ])
        self.l_item_mlp = keras.models.Sequential([
            layers.Dense(n_item_mlp_factors, input_dim=n_users, activation="relu"),
            layers.Dense(n_item_mlp_factors // 2, activation="relu"),
            layers.Dense(n_filters, activation="relu"),
            layers.BatchNormalization(),
        ])
        self.a_user = keras.models.Sequential([
            layers.Dense(attention_size, activation="relu", use_bias=True),
            layers.Dense(1, activation=None, use_bias=True)
        ])
        self.user_attention = layers.Softmax(axis=1, name="user_attention")
        self.a_item = keras.models.Sequential([
            layers.Dense(attention_size, activation="relu", use_bias=True),
            layers.Dense(1, activation=None, use_bias=True)
        ])
        self.item_attention = layers.Softmax(axis=1, name="item_attention")
        self.ou_dropout = layers.Dropout(rate=dropout_rate)
        self.oi_dropout = layers.Dropout(rate=dropout_rate)
        self.ou = layers.Dense(n_factors, use_bias=True, name="ou")
        self.oi = layers.Dense(n_factors, use_bias=True, name="oi")
        self.W1 = layers.Dense(1, activation=None, use_bias=False, name="W1")
        self.add_global_bias = AddGlobalBias(init_value=global_mean, name="global_bias")

    def call(self, inputs, training=False):
        i_user_id, i_item_id, i_user_rating, i_user_review, i_user_num_reviews, i_item_rating, i_item_review, i_item_num_reviews = inputs
        user_review_h = self.user_text_processor(self.l_user_review_embedding(i_user_review), training=training)
        item_review_h = self.item_text_processor(self.l_item_review_embedding(i_item_review), training=training)
        user_rating_h = self.l_user_mlp(i_user_rating)
        item_rating_h = self.l_item_mlp(i_item_rating)
        a_user = self.a_user(
            tf.multiply(
                user_review_h,
                tf.expand_dims(user_rating_h, 1)
            )
        )
        a_user_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_user_num_reviews, [-1]), maxlen=i_user_review.shape[1]), -1)
        user_attention = self.user_attention(a_user, a_user_masking)
        a_item = self.a_item(
            tf.multiply(
                item_review_h,
                tf.expand_dims(item_rating_h, 1)
            )
        )
        a_item_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_item_num_reviews, [-1]), maxlen=i_item_review.shape[1]), -1)
        item_attention = self.item_attention(a_item, a_item_masking)
        ou = tf.multiply(user_attention, user_review_h)
        ou = tf.reduce_sum(ou, 1)
        if training:
            ou = self.ou_dropout(ou, training=training)
        ou = self.ou(ou)
        oi = tf.multiply(item_attention, item_review_h)
        oi = tf.reduce_sum(oi, 1)
        if training:
            oi = self.oi_dropout(oi, training=training)
        oi = self.oi(oi)
        pu = tf.concat([
            user_rating_h,
            ou,
            self.l_user_embedding(i_user_id)
        ], axis=-1)
        qi = tf.concat([
            item_rating_h,
            oi,
            self.l_item_embedding(i_item_id)
        ], axis=-1)
        h0 = tf.multiply(pu, qi)
        r = self.add_global_bias(
            tf.add_n([
                self.W1(h0),
                self.user_bias(i_user_id),
                self.item_bias(i_item_id)
            ])
        )
        return r

class HRDRModel:
    def __init__(self, n_users, n_items, vocab, global_mean,
                 n_factors=32, embedding_size=100, id_embedding_size=32,
                 attention_size=16, kernel_sizes=[3], n_filters=64,
                 n_user_mlp_factors=128, n_item_mlp_factors=128,
                 dropout_rate=0.5, max_text_length=50, max_num_review=32,
                 pretrained_word_embeddings=None, verbose=False, seed=None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_vocab = vocab.size
        self.global_mean = global_mean
        self.n_factors = n_factors
        self.embedding_size = embedding_size
        self.id_embedding_size = id_embedding_size
        self.attention_size = attention_size
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.n_user_mlp_factors = n_user_mlp_factors
        self.n_item_mlp_factors = n_item_mlp_factors
        self.dropout_rate = dropout_rate
        self.max_text_length = max_text_length
        self.max_num_review = max_num_review
        self.verbose = verbose
        if seed is not None:
            self.rng = get_rng(seed)
            tf.random.set_seed(seed)

        embedding_matrix = uniform(shape=(self.n_vocab, self.embedding_size), low=-0.5, high=0.5, random_state=self.rng)
        embedding_matrix[:4, :] = np.zeros((4, self.embedding_size))
        if pretrained_word_embeddings is not None:
            oov_count = 0
            for word, idx in vocab.tok2idx.items():
                embedding_vector = pretrained_word_embeddings.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector
                else:
                    oov_count += 1
            if self.verbose:
                print("Number of OOV words: %d" % oov_count)

        embedding_matrix = initializers.Constant(embedding_matrix)
        self.graph = Model(
            self.n_users, self.n_items, self.n_vocab, self.global_mean, embedding_matrix,
            self.n_factors, self.embedding_size, self.id_embedding_size,
            self.attention_size, self.kernel_sizes, self.n_filters,
            self.n_user_mlp_factors, self.n_item_mlp_factors,
            self.dropout_rate, self.max_text_length
        )

    def get_weights(self, train_set, batch_size=64):
        P = np.zeros((self.n_users, self.n_filters + self.n_factors + self.id_embedding_size))
        Q = np.zeros((self.n_items, self.n_filters + self.n_factors + self.id_embedding_size))
        A = np.zeros((self.n_items, self.max_num_review))
        for batch_users in train_set.user_iter(batch_size, shuffle=False):
            i_user_review, i_user_num_reviews, i_user_rating = get_data(batch_users, train_set, self.max_text_length, by='user', max_num_review=self.max_num_review)
            user_review_embedding = self.graph.l_user_review_embedding(i_user_review)
            user_review_h = self.graph.user_text_processor(user_review_embedding, training=False)
            user_rating_h = self.graph.l_user_mlp(i_user_rating)
            a_user = self.graph.a_user(
                tf.multiply(
                    user_review_h,
                    tf.expand_dims(user_rating_h, 1)
                )
            )
            a_user_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_user_num_reviews, [-1]), maxlen=i_user_review.shape[1]), -1)
            user_attention = self.graph.user_attention(a_user, a_user_masking)
            ou = self.graph.ou(tf.reduce_sum(tf.multiply(user_attention, user_review_h), 1))
            pu = tf.concat([
                user_rating_h,
                ou,
                self.graph.l_user_embedding(batch_users)
            ], axis=-1)
            P[batch_users] = pu.numpy()
        for batch_items in train_set.item_iter(batch_size, shuffle=False):
            i_item_review, i_item_num_reviews, i_item_rating = get_data(batch_items, train_set, self.max_text_length, by='item', max_num_review=self.max_num_review)
            item_review_embedding = self.graph.l_item_review_embedding(i_item_review)
            item_review_h = self.graph.item_text_processor(item_review_embedding, training=False)
            item_rating_h = self.graph.l_item_mlp(i_item_rating)
            a_item = self.graph.a_item(
                tf.multiply(
                    item_review_h,
                    tf.expand_dims(item_rating_h, 1)
                )
            )
            a_item_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_item_num_reviews, [-1]), maxlen=i_item_review.shape[1]), -1)
            item_attention = self.graph.item_attention(a_item, a_item_masking)
            oi = self.graph.oi(tf.reduce_sum(tf.multiply(item_attention, item_review_h), 1))
            qi = tf.concat([
                item_rating_h,
                oi,
                self.graph.l_item_embedding(batch_items)
            ], axis=-1)
            Q[batch_items] = qi.numpy()
            A[batch_items, :item_attention.shape[1]] = item_attention.numpy().reshape(item_attention.shape[:2])
        W1 = self.graph.W1.get_weights()[0]
        bu = self.graph.user_bias.get_weights()[0]
        bi = self.graph.item_bias.get_weights()[0]
        mu = self.graph.add_global_bias.get_weights()[0][0]
        return P, Q, W1, bu, bi, mu, A
