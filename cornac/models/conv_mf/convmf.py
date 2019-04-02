"""
@author: Tran Thanh Binh

"""
import time
import math
import numpy as np
from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    tf = None


def convmf(train_set, give_item_weight=True,
           max_iter=50, lambda_u=1, lambda_v=100,
           init_params=None, dimension=50, vocab_size=8000,
           dropout_rate=0.2, emb_dim=200, max_len=300, num_kernel_per_ws=100):
    endure = 3
    converge_threshold = 0.01
    history = 1e-50
    loss = 0

    R_user = train_set.matrix
    user = []

    user_index_list = []
    user_rating_list = []
    for i in range(R_user.shape[0]):
        item_idx = R_user[i].nonzero()[1]
        rating = R_user[i, item_idx].A[0]
        user_index_list.append(item_idx)
        user_rating_list.append(rating)

    user.append(user_index_list)
    user.append(user_rating_list)

    R_item = train_set.matrix.tocsc().T
    item = []

    item_index_list = []
    item_rating_list = []
    for i in range(R_item.shape[0]):
        user_idx = R_item[i].nonzero()[1]
        rating = R_item[i, user_idx].A[0]
        item_index_list.append(user_idx)
        item_rating_list.append(rating)

    item.append(item_index_list)
    item.append(item_rating_list)

    n_user = len(user[0])
    n_item = len(item[0])

    # R_user and R_item contain rating values
    R_user = user[1]
    R_item = item[1]

    if give_item_weight:
        item_weight = np.array([math.sqrt(len(i))
                                for i in R_item], dtype=float)
        item_weight = (float(n_item) / item_weight.sum()) * item_weight
    else:
        item_weight = np.ones(n_item, dtype=float)

    # Initialize cnn module
    cnn_module = CNN_module(output_dimesion=dimension, vocab_size=vocab_size,
                            dropout_rate=dropout_rate, emb_dim=emb_dim,
                            max_len=max_len, nb_filters=num_kernel_per_ws, init_W=init_params.get('W', None))

    # initialize U and V matrix
    U = init_params.get('U', np.random.uniform(size=(n_user, dimension)))
    V = init_params.get('V', np.random.uniform(size=(n_item, dimension)))

    document = train_set.item_text.batch_seq(np.arange(n_item), max_length=max_len)
    theta = cnn_module.get_projection_layer(document)

    for iter in range(max_iter):
        print("Iteration {}".format(iter + 1))
        tic = time.time()

        user_loss = np.zeros(n_user)
        for i in range(n_user):
            idx_item = user[0][i]
            V_i = V[idx_item]
            R_i = R_user[i]

            A = lambda_u * np.eye(dimension) + _square(V_i)
            B = (V_i * (np.tile(R_i, (dimension, 1)).T)).sum(0)
            U[i] = np.linalg.solve(A, B)

            user_loss[i] = -0.5 * lambda_u * np.dot(U[i], U[i])

        item_loss = np.zeros(n_item)
        for j in range(n_item):
            idx_user = item[0][j]
            U_j = U[idx_user]
            R_j = R_item[j]
            Uj_square = _square(U_j)

            A = lambda_v * item_weight[j] * np.eye(dimension) + Uj_square
            B = (U_j * (np.tile(R_j, (dimension, 1)).T)).sum(0) \
                + lambda_v * item_weight[j] * theta[j]
            V[j] = np.linalg.solve(A, B)

            item_loss[j] = -0.5 * np.square(R_j).sum()
            item_loss[j] = item_loss[j] + np.sum((U_j.dot(V[j])) * R_j)
            item_loss[j] = item_loss[j] - 0.5 * np.dot(V[j].dot(Uj_square), V[j])

        cnn_loss = cnn_module.train(train_set=train_set, V=V, item_weight=item_weight)
        theta = cnn_module.get_projection_layer(X_train=document)

        loss = loss + np.sum(user_loss) + np.sum(item_loss) - 0.5 * lambda_v * cnn_loss * n_item

        toc = time.time()
        elapsed = toc - tic
        converge = abs((loss - history) / history)
        print("Loss: %.5f Elpased: %.4fs Converge: %.6f " % (loss, elapsed, converge))
        history = loss

        if converge < converge_threshold:
            endure -= 1
            if endure == 0:
                break

    return {'U': U, 'V': V}


def _square(mat):
    """
    return XT.X matrix
    """
    return mat.T.dot(mat)


class CNN_module():

    def _new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def _new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def _convlayer(self, input, num_input_channels,
                   filter_height, filter_width,
                   num_filters, use_pooling=True):

        shape = [filter_height, filter_width, num_input_channels, num_filters]
        weights = self._new_weights(shape)
        biases = self._new_biases(num_filters)
        layer = tf.nn.conv2d(input=input, filter=weights,
                             strides=[1, 1, 1, 1], padding="VALID")
        layer = layer + biases
        if use_pooling:
            layer = tf.nn.max_pool(value=layer, ksize=[1, input.shape[1] - filter_height + 1, 1, 1],
                                   strides=[1, 1, 1, 1], padding="VALID")
        layer = tf.nn.relu(layer)
        return layer, weights

    def _flatten_layer(self, layer):

        layer_shape = layer.get_shape()
        num_feature = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_feature])
        return layer_flat, num_feature

    def _fc_layer(self, input, num_input, num_output):

        weights = self._new_weights(shape=[num_input, num_output])
        biases = self._new_biases(length=num_output)
        layer = tf.matmul(input, weights) + biases
        layer = tf.nn.tanh(layer)
        return layer

    def __init__(self, output_dimesion, vocab_size,
                 dropout_rate, emb_dim, max_len, nb_filters,
                 batch_size=128, nb_epoch=5, init_W=None):

        self.nb_epoch = nb_epoch
        self.drop_rate = dropout_rate
        self.batch_size = batch_size
        self.sess = tf.Session()  # Tensorflow session
        self.max_len = max_len

        vanila_dimension = 200
        learning_rate = 0.001
        filter_lengths = [3, 4, 5]

        # create Graph
        self.model_input = tf.placeholder(dtype=tf.int32, shape=(None, max_len))
        self.v = tf.placeholder(dtype=tf.float32, shape=(None, output_dimesion))
        self.sample_weight = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.droprate_holder = tf.placeholder_with_default(1.0, shape=())

        # Embedding layer
        if init_W is None:
            self.embedding_weight = tf.Variable(tf.random_uniform([vocab_size, emb_dim], -1.0, 1.0))
        else:
            self.embedding_weight = tf.convert_to_tensor(init_W)

        self.seq_emb = tf.nn.embedding_lookup(self.embedding_weight, self.model_input)
        self.reshape = tf.reshape(self.seq_emb, [-1, max_len, emb_dim, 1])
        self.convs = []

        # Convolutional layer
        for i in filter_lengths:
            conv_layer, weights = self._convlayer(input=self.reshape, num_input_channels=1,
                                                  filter_height=i, filter_width=emb_dim,
                                                  num_filters=nb_filters, use_pooling=True)

            flat_layer, _ = self._flatten_layer(conv_layer)
            self.convs.append(flat_layer)

        self.model_output = tf.concat(self.convs, axis=-1)
        # Fully-connected layers
        self.model_output = self._fc_layer(input=self.model_output, num_input=self.model_input.get_shape()[1].value,
                                           num_output=vanila_dimension)
        # Dropout layer
        self.model_output = tf.nn.dropout(self.model_output, self.drop_rate)
        # Output layer
        self.model_output = self._fc_layer(input=self.model_output, num_input=vanila_dimension,
                                           num_output=output_dimesion)
        # Weighted MEA loss function
        self.mean_square_loss = tf.losses.mean_squared_error(labels=self.v, predictions=self.model_output,
                                                             reduction=tf.losses.Reduction.NONE)
        self.weighted_loss = tf.reduce_sum(
            tf.reduce_sum(self.mean_square_loss, axis=1, keepdims=True) * self.sample_weight)
        # RMSPro optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.weighted_loss)

        self.sess.run(tf.global_variables_initializer())  # init variable

    def train(self, train_set, V, item_weight):
        for _ in tqdm(range(self.nb_epoch), desc='CNN'):
            for batch_ids in train_set.item_iter(batch_size=self.batch_size, shuffle=True):
                batch_seq = train_set.item_text.batch_seq(batch_ids, max_length=self.max_len)
                feed_dict = {self.model_input: batch_seq,
                             self.droprate_holder: self.drop_rate,
                             self.v: V[batch_ids],
                             self.sample_weight: item_weight[batch_ids]}

                _, history = self.sess.run([self.optimizer, self.weighted_loss], feed_dict=feed_dict)

        return history

    def get_projection_layer(self, X_train):

        feed_dict = {self.model_input: X_train}
        prediction = self.sess.run([self.model_output], feed_dict=feed_dict)
        return np.array(prediction).reshape(len(X_train), -1)
