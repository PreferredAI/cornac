"""
@author: Tran Thanh Binh
"""

import numpy as np
from operator import itemgetter
from ....utils.data_utils import Dataset
from ....utils import tryimport
tf = tryimport('tensorflow')

class CNN_module():

  def _new_weights(self,shape):
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
    layer = layer+biases
    if use_pooling:
      layer = tf.nn.max_pool(value=layer, ksize=[1, input.shape[1]-filter_height+1, 1, 1],
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

    vanila_dimension = 200
    learning_rate = 0.001
    filter_lengths = [3, 4, 5]

    #create Graph
    self.model_input = tf.placeholder(dtype=tf.int32, shape=(None, max_len))
    self.v = tf.placeholder(dtype=tf.float32, shape=(None, output_dimesion))
    self.sample_weight = tf.placeholder(dtype=tf.float32, shape=(None, ))
    self.droprate_holder = tf.placeholder_with_default(1.0, shape=())

    #Embedding layer
    if init_W is None:
      self.embedding_weight = tf.Variable(tf.random_uniform([vocab_size, emb_dim], -1.0, 1.0))
    else:
      self.embedding_weight = tf.convert_to_tensor(init_W)

    self.seq_emb = tf.nn.embedding_lookup(self.embedding_weight, self.model_input)
    self.reshape = tf.reshape(self.seq_emb, [-1,max_len, emb_dim,1])
    self.convs = []

    #Convolutional layer
    for i in filter_lengths:
      conv_layer, weights = self._convlayer(input=self.reshape, num_input_channels=1,
                      filter_height=i, filter_width=emb_dim,
                      num_filters=nb_filters, use_pooling=True)

      flat_layer, _ = self._flatten_layer(conv_layer)
      self.convs.append(flat_layer)

    self.model_output = tf.concat(self.convs, axis=-1)
    #Fully-connected layers
    self.model_output = self._fc_layer(input=self.model_output, num_input=self.model_input.get_shape()[1].value,
                                  num_output=vanila_dimension)
    #Dropout layer
    self.model_output = tf.nn.dropout(self.model_output, self.drop_rate)
    #Output layer
    self.model_output = self._fc_layer(input=self.model_output, num_input=vanila_dimension,
                                  num_output=output_dimesion)
    #Weighted MEA loss function
    self.mean_square_loss = tf.losses.mean_squared_error(labels=self.v, predictions=self.model_output,
                                                         reduction=tf.losses.Reduction.NONE)
    self.weighted_loss = tf.reduce_sum(tf.reduce_sum(self.mean_square_loss, axis=1, keepdims=True)*self.sample_weight)
    #RMSPro optimizer
    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.weighted_loss)

    self.sess.run(tf.global_variables_initializer()) #init variable

  def train(self, X_train, V, item_weight):

    print("Train...CNN module")
    for _ in range(self.nb_epoch):
      indx = Dataset(np.random.permutation(len(X_train)))
      num_steps = int(len(X_train) / self.batch_size)
      for i in range(1, num_steps + 1):
        batch, _ = indx.next_batch(self.batch_size)
        feed_dict = {self.model_input: itemgetter(*batch)(X_train), self.droprate_holder: self.drop_rate,
                     self.v: itemgetter(*batch)(V), self.sample_weight:itemgetter(*batch)(item_weight)}

        _, history = self.sess.run([self.optimizer, self.weighted_loss], feed_dict=feed_dict)

    return history

  def get_projection_layer(self, X_train):

    feed_dict = {self.model_input: X_train}
    prediction = self.sess.run([self.model_output], feed_dict=feed_dict)
    return np.array(prediction).reshape(len(X_train),-1)