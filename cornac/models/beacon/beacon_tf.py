import numpy as np
import warnings

# disable annoying tensorflow deprecated API warnings
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()


def create_rnn_cell(cell_type, state_size, default_initializer, reuse=None):
    if cell_type == "GRU":
        return tf.nn.rnn_cell.GRUCell(state_size, activation=tf.nn.tanh, reuse=reuse)
    elif cell_type == "LSTM":
        return tf.nn.rnn_cell.LSTMCell(
            state_size,
            initializer=default_initializer,
            activation=tf.nn.tanh,
            reuse=reuse,
        )
    else:
        return tf.nn.rnn_cell.BasicRNNCell(
            state_size, activation=tf.nn.tanh, reuse=reuse
        )


def create_rnn_encoder(
    x,
    rnn_units,
    dropout_rate,
    seq_length,
    rnn_cell_type,
    param_initializer,
    seed,
    reuse=None,
):
    with tf.variable_scope("RNN_Encoder", reuse=reuse):
        rnn_cell = create_rnn_cell(rnn_cell_type, rnn_units, param_initializer)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell, input_keep_prob=1 - dropout_rate, seed=seed
        )
        init_state = rnn_cell.zero_state(tf.shape(x)[0], tf.float32)
        # RNN Encoder: Iteratively compute output of recurrent network
        rnn_outputs, _ = tf.nn.dynamic_rnn(
            rnn_cell,
            x,
            initial_state=init_state,
            sequence_length=seq_length,
            dtype=tf.float32,
        )
        return rnn_outputs


def create_basket_encoder(
    x,
    dense_units,
    param_initializer,
    activation_func=None,
    name="Basket_Encoder",
    reuse=None,
):
    with tf.variable_scope(name, reuse=reuse):
        return tf.layers.dense(
            x,
            dense_units,
            kernel_initializer=param_initializer,
            bias_initializer=tf.zeros_initializer,
            activation=activation_func,
        )


def get_last_right_output(full_output, max_length, actual_length, rnn_units):
    batch_size = tf.shape(full_output)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * max_length + (actual_length - 1)
    # Indexing
    return tf.gather(tf.reshape(full_output, [-1, rnn_units]), index)


class BeaconModel:
    def __init__(
        self,
        sess,
        emb_dim,
        rnn_units,
        alpha,
        max_seq_length,
        n_items,
        item_probs,
        adj_matrix,
        rnn_cell_type,
        rnn_dropout_rate,
        seed,
        lr,
    ):
        self.scope = "GRN"
        self.session = sess
        self.seed = seed
        self.lr = tf.constant(lr)

        self.emb_dim = emb_dim
        self.rnn_units = rnn_units

        self.max_seq_length = max_seq_length
        self.n_items = n_items
        self.item_probs = item_probs
        self.alpha = alpha

        with tf.variable_scope(self.scope):
            # Initialized for n_hop adjacency matrix
            self.A = tf.constant(
                adj_matrix.todense(), name="Adj_Matrix", dtype=tf.float32
            )

            uniform_initializer = (
                np.ones(shape=(self.n_items), dtype=np.float32) / self.n_items
            )
            self.I_B = tf.get_variable(
                dtype=tf.float32,
                initializer=tf.constant(uniform_initializer, dtype=tf.float32),
                name="I_B",
            )
            self.I_B_Diag = tf.nn.relu(tf.diag(self.I_B, name="I_B_Diag"))

            self.C_Basket = tf.get_variable(
                dtype=tf.float32, initializer=tf.constant(adj_matrix.mean()), name="C_B"
            )
            self.y = tf.placeholder(
                dtype=tf.float32,
                shape=(None, self.n_items),
                name="Target_basket",
            )

            # Basket Sequence encoder
            with tf.name_scope("Basket_Sequence_Encoder"):
                self.bseq = tf.sparse.placeholder(
                    dtype=tf.float32,
                    name="bseq_input",
                )
                self.bseq_length = tf.placeholder(
                    dtype=tf.int32, shape=(None,), name="bseq_length"
                )

                self.bseq_encoder = tf.sparse.reshape(
                    self.bseq, shape=[-1, self.n_items], name="bseq_2d"
                )
                self.bseq_encoder = self.encode_basket_graph(
                    self.bseq_encoder, self.C_Basket, True
                )
                self.bseq_encoder = tf.reshape(
                    self.bseq_encoder,
                    shape=[-1, self.max_seq_length, self.n_items],
                    name="bsxMxN",
                )
                self.bseq_encoder = create_basket_encoder(
                    self.bseq_encoder,
                    emb_dim,
                    param_initializer=tf.initializers.he_uniform(),
                    activation_func=tf.nn.relu,
                )

                # batch_size x max_seq_length x H
                rnn_encoder = create_rnn_encoder(
                    self.bseq_encoder,
                    self.rnn_units,
                    rnn_dropout_rate,
                    self.bseq_length,
                    rnn_cell_type,
                    param_initializer=tf.initializers.glorot_uniform(),
                    seed=self.seed,
                )

                # Hack to build the indexing and retrieve the right output. # batch_size x H
                h_T = get_last_right_output(
                    rnn_encoder, self.max_seq_length, self.bseq_length, self.rnn_units
                )

            # Next basket estimation
            with tf.name_scope("Next_Basket"):
                W_H = tf.get_variable(
                    dtype=tf.float32,
                    initializer=tf.initializers.glorot_uniform(),
                    shape=(self.rnn_units, self.n_items),
                    name="W_H",
                )

                next_item_probs = tf.nn.sigmoid(tf.matmul(h_T, W_H))
                logits = (
                    1.0 - self.alpha
                ) * next_item_probs + self.alpha * self.encode_basket_graph(
                    next_item_probs, tf.constant(0.0)
                )

            with tf.name_scope("Loss"):
                self.loss = self.compute_loss(logits, self.y)

                self.predictions = tf.nn.sigmoid(logits)

            # Adam optimizer
            train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr)

            # Op to calculate every variable gradient
            self.grads = train_op.compute_gradients(self.loss, tf.trainable_variables())
            self.update_grads = train_op.apply_gradients(self.grads)

    def train_batch(self, s, s_length, y):
        bseq_indices, bseq_values, bseq_shape = self.get_sparse_tensor_info(s, True)

        [_, loss] = self.session.run(
            [self.update_grads, self.loss],
            feed_dict={
                self.bseq: (bseq_indices, bseq_values, bseq_shape),
                self.bseq_length: s_length,
                self.y: y,
            },
        )

        return loss

    def validate_batch(self, s, s_length, y):
        bseq_indices, bseq_values, bseq_shape = self.get_sparse_tensor_info(s, True)

        loss = self.session.run(
            self.loss,
            feed_dict={
                self.bseq: (bseq_indices, bseq_values, bseq_shape),
                self.bseq_length: s_length,
                self.y: y,
            },
        )
        return loss

    def predict(self, s, s_length):
        bseq_indices, bseq_values, bseq_shape = self.get_sparse_tensor_info(s, True)
        predictions = self.session.run(
            self.predictions,
            feed_dict={
                self.bseq: (bseq_indices, bseq_values, bseq_shape),
                self.bseq_length: s_length,
            },
        )
        return predictions.squeeze()

    def encode_basket_graph(self, binput, beta, is_sparse=False):
        with tf.name_scope("Graph_Encoder"):
            if is_sparse:
                encoder = tf.sparse_tensor_dense_matmul(
                    binput, self.I_B_Diag, name="XxI_B"
                )
                encoder += self.relu_with_threshold(
                    tf.sparse_tensor_dense_matmul(binput, self.A, name="XxA"), beta
                )
            else:
                encoder = tf.matmul(binput, self.I_B_Diag, name="XxI_B")
                encoder += self.relu_with_threshold(
                    tf.matmul(binput, self.A, name="XxA"), beta
                )
        return encoder

    def get_sparse_tensor_info(self, x, is_bseq=False):
        indices = []
        if is_bseq:
            for sid, bseq in enumerate(x):
                for t, basket in enumerate(bseq):
                    for item_id in basket:
                        indices.append([sid, t, item_id])
        else:
            for bid, basket in enumerate(x):
                for item_id in basket:
                    indices.append([bid, item_id])

        values = np.ones(len(indices), dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)
        shape = np.array([len(x), self.max_seq_length, self.n_items], dtype=np.int64)
        return indices, values, shape

    def compute_loss(self, logits, y):
        sigmoid_logits = tf.nn.sigmoid(logits)

        neg_y = 1.0 - y
        pos_logits = y * logits

        pos_max = tf.reduce_max(pos_logits, axis=1)
        pos_max = tf.expand_dims(pos_max, axis=-1)

        pos_min = tf.reduce_min(pos_logits + neg_y * pos_max, axis=1)
        pos_min = tf.expand_dims(pos_min, axis=-1)

        nb_pos, nb_neg = tf.count_nonzero(y, axis=1), tf.count_nonzero(neg_y, axis=1)
        ratio = tf.cast(nb_neg, dtype=tf.float32) / tf.cast(nb_pos, dtype=tf.float32)

        pos_weight = tf.expand_dims(ratio, axis=-1)
        loss = y * -tf.log(sigmoid_logits) * pos_weight + neg_y * -tf.log(
            1.0 - tf.nn.sigmoid(logits - pos_min)
        )

        return tf.reduce_mean(loss + 1e-8)

    def relu_with_threshold(self, x, threshold):
        return tf.nn.relu(x - tf.abs(threshold))
