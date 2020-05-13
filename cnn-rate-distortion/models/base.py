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


from abc import ABC, abstractmethod
import os
import sys
import time

import h5py
import numpy as np
import tensorflow as tf


class BaseModel(ABC):
    def __init__(self, model_name, session, epochs, batch_size, loss_type, width, height, levels):
        self._model_name = model_name
        self._session = session
        self._epochs = epochs
        self._batch_size = batch_size
        self._loss_type = loss_type

        self._width = width
        self._height = height
        self._channels = 1
        self._levels = levels

        self._input = None
        self._label = None
        self._output = None

        self._loss = None
        self._loss_summary = None

        self._variables = {}
        self._train_var = None
        self._summary = None
        self._saver = None
        self._writer = None
        self._build()

    @abstractmethod
    def _set_placeholders(self):
        pass

    @abstractmethod
    def cnn(self):
        pass

    @abstractmethod
    def _shuffle_dataset(self, dataset_, train_batch_order, batch):
        pass

    def _set_loss(self):
        if self._loss_type == 'MSE':
            self._loss = tf.reduce_mean(tf.squared_difference(self._label, self._output))
        elif self._loss_type == 'MAE':
            self._loss = tf.reduce_mean(tf.losses.absolute_difference(self._label, self._output))
        else:
            self._loss = tf.reduce_mean(tf.squared_difference(tf.log(self._label + 1.), tf.log(self._output + 1.)))

        self._loss_summary = tf.summary.scalar("LossSummary", self._loss)

    def _build(self):
        self._set_placeholders()
        self._output = self.cnn()
        self._set_loss()
        self._train_var = tf.trainable_variables()
        self._saver = tf.train.Saver()

    @staticmethod
    def _prelu(_x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.25), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def load(self, archive_dir):
        archive_dir = os.path.join(archive_dir, self._model_name)
        checkpoint = tf.train.get_checkpoint_state(archive_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            self._saver.restore(self._session, os.path.join(archive_dir, checkpoint_name))
            print("[!] Network loaded.")
        else:
            print("[x] No records found.")

    def save(self, archive_dir, counter):
        archive_dir = os.path.join(archive_dir, self._model_name)

        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)

        self._saver.save(self._session, os.path.join(archive_dir, self._model_name), global_step=counter)
        print("[*] Model saved to archives.")
        
    def train(self, config):
        file_name = '{0}_{1}_{2}x{3}.h5'.format(config.data_name, self._model_name, self._width, self._height)
        file_name = os.path.join(config.h5_dir, file_name)

        with h5py.File(file_name, 'r') as dataset_:
            total_dataset_size = dataset_['input'].shape[0] - 1
            training_size = int(total_dataset_size * .8)

            for w in self._train_var:
                self._loss += tf.nn.l2_loss(w) * config.weight_decay

            global_step = tf.Variable(0, trainable=False)

            train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self._loss, global_step=global_step)
            tf.global_variables_initializer().run()

            self._saver = tf.train.Saver(self._train_var)
            self._summary = tf.summary.merge([self._loss_summary])
            self._writer = tf.summary.FileWriter("./logs", self._session.graph)

            number_of_batches = int(training_size / self._batch_size)
            train_batch_order = np.random.permutation(np.arange(training_size)) + 1

            start_time = time.time()
            val_loss = sys.float_info.max
            num_of_no_improves = 0

            self.load(config.archive_dir)

            for epoch in range(self._epochs):
                for i in range(number_of_batches):
                    feed_dict = self._shuffle_dataset(dataset_, train_batch_order, i)
                    _, summary_str, error = self._session.run(
                        [train_op, self._summary, self._loss], feed_dict=feed_dict)

                    if global_step.eval() % 100 == 0:
                        self._writer.add_summary(summary_str, global_step.eval())
                        print("Epoch: %4d; Iteration: %10d; Batch: %4d/%4d; LR: %.8f; Time: %4.4f; Loss: %.8f"
                              % (epoch, global_step.eval(), i, number_of_batches, config.learning_rate,
                                 time.time() - start_time, error))

                tmp_val_loss = self.eval(dataset_)
                print("--- Validation; Epoch: %4d; AVG loss: %.8f" % (epoch, tmp_val_loss))

                if tmp_val_loss < val_loss:
                    val_loss = tmp_val_loss
                    num_of_no_improves = 0
                    self.save(config.archive_dir, global_step.eval())
                else:
                    num_of_no_improves += 1

                if num_of_no_improves == 50:
                    print('Total time %4.4f' % (time.time() - start_time))
                    print("Early stopping...")
                    break

    def eval(self, dataset_):
        total_dataset_size = dataset_['input'].shape[0] - 1
        training_size = int(total_dataset_size * .8)
        validation_size = int(total_dataset_size * .2)

        number_of_batches = int(validation_size / self._batch_size)
        loss_eval = np.empty(number_of_batches)

        for i in range(number_of_batches):
            feed_dict = self._shuffle_dataset(dataset_, range(training_size, total_dataset_size), i)
            loss_eval[i] = self._session.run(self._loss, feed_dict=feed_dict)

        return np.mean(loss_eval)

    def test(self, file_name):
        with h5py.File(file_name, 'r') as dataset_:
            total_dataset_size = dataset_['input'].shape[0] - 1
            number_of_batches = int(total_dataset_size / self._batch_size)
            loss_test = np.empty(number_of_batches)

            for i in range(number_of_batches):
                feed_dict = self._shuffle_dataset(dataset_, range(total_dataset_size), i)
                loss_test[i] = self._session.run(self._loss, feed_dict=feed_dict)
                print('Batch: {0}; Loss: {1}'.format(i, loss_test[i]))

            return np.mean(loss_test)
