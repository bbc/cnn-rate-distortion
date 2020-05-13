#
# Copyright 2020 BBC Research & Development
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
#

from models.base import BaseModel

import numpy as np
import tensorflow as tf


class RateModel(BaseModel):
    def __init__(self, model_name, session, epochs, batch_size, loss_type, width, height, levels):
        super().__init__(model_name, session, epochs, batch_size, loss_type, width, height, levels)

    def _set_placeholders(self):
        self._input = tf.placeholder(tf.float32, [None, self._height, self._width, self._channels], name='input')
        self._label = tf.placeholder(tf.float32, [None, self._levels], name='label')
        self._output = tf.placeholder(tf.float32, [None, self._levels], name='output')

    def cnn(self):
        kernel_1 = tf.get_variable(
            'w_1', [3, 3, self._channels, 64], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_1 = tf.get_variable('b_1', [64], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_i = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(input=self._input, filter=kernel_1, strides=[1, 1, 1, 1], padding='SAME'), bias_1))

        kernel_2 = tf.get_variable(
            'w_2', [3, 3, 64, 64], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_2 = tf.get_variable('b_2', [64], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_i = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=conv_i, filter=kernel_2, strides=[1, 1, 1, 1], padding='SAME'), bias_2))

        conv_j = tf.nn.max_pool(value=conv_i, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        kernel_3 = tf.get_variable(
            'w_3', [3, 3, 64, 64], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_3 = tf.get_variable('b_3', [64], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_j = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=conv_j, filter=kernel_3, strides=[1, 1, 1, 1], padding='SAME'), bias_3))

        kernel_4 = tf.get_variable(
            'w_4', [3, 3, 64, 64], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_4 = tf.get_variable('b_4', [64], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_j = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=conv_j, filter=kernel_4, strides=[1, 1, 1, 1], padding='SAME'), bias_4))

        conv_k = tf.nn.max_pool(value=conv_j, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        kernel_5 = tf.get_variable(
            'w_5', [3, 3, 64, 64], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_5 = tf.get_variable('b_5', [64], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_k = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=conv_k, filter=kernel_5, strides=[1, 1, 1, 1], padding='SAME'), bias_5),
            'ReLU_5')

        conv_k = tf.image.resize_images(conv_k, [int(self._height / 2), int(self._width / 2)])

        kernel_6 = tf.get_variable(
            'w_6', [3, 3, 64, 64], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_6 = tf.get_variable('b_6', [64], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_k = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=conv_k, filter=kernel_6, strides=[1, 1, 1, 1], padding='SAME'), bias_6),
            'ReLU_6')

        kernel_7 = tf.get_variable(
            'w_7', [3, 3, 64, 64], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_7 = tf.get_variable('b_7', [64], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_k = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=conv_k, filter=kernel_7, strides=[1, 1, 1, 1], padding='SAME'), bias_7))

        conv_j = tf.concat(values=[conv_j, conv_k], axis=3)

        conv_j = tf.image.resize_images(conv_j, [self._height, self._width])

        kernel_8 = tf.get_variable(
            'w_8', [3, 3, 128, 64], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_8 = tf.get_variable('b_8', [64], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_j = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=conv_j, filter=kernel_8, strides=[1, 1, 1, 1], padding='SAME'), bias_8))

        kernel_9 = tf.get_variable(
            'w_9', [3, 3, 64, 64], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_9 = tf.get_variable('b_9', [64], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_j = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=conv_j, filter=kernel_9, strides=[1, 1, 1, 1], padding='SAME'), bias_9))

        conv_i = tf.concat(values=[conv_i, conv_j], axis=3)

        kernel_10 = tf.get_variable(
            'w_10', [3, 3, 128, 64], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_10 = tf.get_variable('b_10', [64], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_i = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=conv_i, filter=kernel_10, strides=[1, 1, 1, 1], padding='SAME'), bias_10),
            'ReLU_10')

        kernel_11 = tf.get_variable(
            'w_11', [3, 3, 64, 1], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_11 = tf.get_variable('b_11', [1], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_i = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input=conv_i, filter=kernel_11, strides=[1, 1, 1, 1], padding='SAME'), bias_11),
            'ReLU_11')

        conv_i = tf.reshape(conv_i, [-1, self._width * self._height])

        fc_weight_1 = tf.get_variable(
            'w_12', [self._width * self._height, 128], tf.float32,
            initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_12 = tf.get_variable('b_12', [128], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_i = tf.nn.relu(tf.nn.bias_add(tf.matmul(conv_i, fc_weight_1), bias_12))

        fc_weight_2 = tf.get_variable(
            'w_13', [128, self._levels], tf.float32, initializer=tf.random_normal_initializer(stddev=0.1179))
        bias_13 = tf.get_variable('b_13', [self._levels], tf.float32, initializer=tf.constant_initializer(0.0))
        conv_i = tf.nn.bias_add(tf.matmul(conv_i, fc_weight_2), bias_13)

        return conv_i

    def _shuffle_dataset(self, dataset_, train_batch_order, batch):
        input_t = []
        label_t = []

        for i in range(self._batch_size):
            input_t.append(dataset_['input'][train_batch_order[i + batch * self._batch_size], :, :, :])
            label_t.append(dataset_['label'][train_batch_order[i + batch * self._batch_size], :])

        feed_dict = {self._input: np.array(input_t), self._label: np.array(label_t) / dataset_.attrs['global_max']}
        return feed_dict
