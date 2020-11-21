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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers


class TextProcessor(Model):
    def __init__(self, max_review_length, filters=64, kernel_sizes=[3], dropout_rate=0.5):
        super(TextProcessor, self).__init__()
        self.max_review_length = max_review_length
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.conv = []
        self.maxpool = []
        for kernel_size in kernel_sizes:
            self.conv.append(layers.Conv2D(self.filters, kernel_size=(1, kernel_size), use_bias=True, activation="relu"))
            self.maxpool.append(layers.MaxPooling2D(pool_size=(1, self.max_review_length - kernel_size + 1)))
        self.reshape = layers.Reshape(target_shape=(-1, self.filters * len(kernel_sizes)))
        self.dropout_rate = dropout_rate
        self.dropout = layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs, training=False):
        review = inputs
        pooled_review_outputs = []
        for conv, maxpool in zip(self.conv, self.maxpool):
            review_conv = conv(review)
            review_conv_maxpool = maxpool(review_conv)
            pooled_review_outputs.append(review_conv_maxpool)
        review_h = self.reshape(tf.concat(pooled_review_outputs, axis=-1))
        if training:
            review_h = self.dropout(review_h)
        return review_h

