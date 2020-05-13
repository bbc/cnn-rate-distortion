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


from models.distortion import DistortionModel
from models.rate import RateModel
from tools.dataset import prepare_distortion_data, prepare_rate_data

import os
import sys

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_float('gpu_fraction', 0.6, 'Fraction of GPU usage')
flags.DEFINE_string('task', 'train', 'Mode of the function')
flags.DEFINE_string('model_name', 'distortion', 'Network architecture "distortion" or "rate"')
flags.DEFINE_integer('epochs', 80000, 'Epochs to train [80000]')
flags.DEFINE_integer('batch_size', 128, 'Size of each training batch [128]')
flags.DEFINE_string('loss_type', 'MSE', 'Loss function used [L2 regularised MSE]')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate [1e-4]')
flags.DEFINE_float('weight_decay', 0.0001, 'The weight decay [1e-4]')
flags.DEFINE_string('archive_dir', './archive', 'Directory to save the training progress')

flags.DEFINE_integer('width', 128, 'Width [128]')
flags.DEFINE_integer('height', 128, 'Height [128]')
flags.DEFINE_integer('levels', 4, 'Number of levels [4]')

flags.DEFINE_string('data_name', 'MSCOCO', 'Dataset name')
flags.DEFINE_string('h5_dir', './dataset', 'Directory to save the dataset')
flags.DEFINE_string('input_dir', './original', 'Directory containing input data (yuv)')
flags.DEFINE_string('label_dir', './reconstruction', 'Directory containing label data (yuv or txt)')
flags.DEFINE_string('test_data', './test.h5', 'Path to h5 test dataset')

f = flags.FLAGS


def main():
    if f.task == 'prepare_data':
        params = (f.data_name, f.input_dir, f.label_dir, f.width, f.height, f.levels, f.model_name, f.h5_dir)
        if f.model_name == 'distortion':
            prepare_distortion_data(*params)
        elif f.model_name == 'rate':
            prepare_rate_data(*params)
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=f.gpu_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            params = (f.model_name, session, f.epochs, f.batch_size, f.loss_type, f.width, f.height, f.levels)
            if f.model_name == 'distortion':
                model = DistortionModel(*params)
            elif f.model_name == 'rate':
                model = RateModel(*params)

            if f.task == 'train':
                if not os.path.exists(f.archive_dir):
                    os.makedirs(f.archive_dir)
                model.train(f)
            elif f.task == 'test':
                model.load(f.archive_dir)
                model.test(f.test_data)
    sys.exit()


if __name__ == '__main__':
    main()
