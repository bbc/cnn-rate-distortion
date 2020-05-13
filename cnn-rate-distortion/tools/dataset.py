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


from tools.reader import YuvReader

from glob import glob
import os
import re

import h5py
import numpy as np


def shape_from_filename(filename):
    rgx = re.compile(r'([0-9]+)x([0-9]+)')
    result = re.search(rgx, filename)
    width = int(result.group(1))
    height = int(result.group(2))
    return width, height


def qp_from_file_name(filename):
    rgx = re.compile(r'QP_([0-9]+)')
    result = re.search(rgx, filename)
    qp = float(result.group(1))
    return qp


def read_single_frame(file_name, width, height, format_='yuv420p'):
    with YuvReader(file_name, width, height, format_) as yuv_reader:
        y, _, _ = yuv_reader.next_y_u_v()
        y = y.astype('float')
    return y


def prepare_distortion_data(data_name, orig_dir, reco_dir, width, height, levels, model, h5_dir):
    file_name = '{0}_{1}_{2}x{3}.h5'.format(data_name, model, width, height)
    file_name = os.path.join(h5_dir, file_name)

    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset('input', (1, height, width, 2), maxshape=(None, height, width, 2))
        hf.create_dataset('label', (1, height, width, 1), maxshape=(None, height, width, 1))

        orig_frames = sorted(glob(os.path.join(orig_dir, '*.yuv')))
        reco_frames = sorted(glob(os.path.join(reco_dir, '**', '*.yuv')))

        img_scale = 2 ** 8 - 1.
        qp_scale = 51.
        idx = 0

        for i in range(len(orig_frames)):
            w, h = shape_from_filename(orig_frames[i])
            y_orig = read_single_frame(orig_frames[i], w, h)

            for j in range(levels):
                reco_name = reco_frames[i * levels + j]
                qp = qp_from_file_name(reco_name) / qp_scale
                y_reco = read_single_frame(reco_name, w, h)

                w = width * (w // width)
                h = height * (h // height)

                y_orig = y_orig[:h, :w] / img_scale
                y_reco = y_reco[:h, :w] / img_scale

                diff = np.abs(y_orig - y_reco)

                for y in range(0, h, height):
                    for x in range(0, w, width):
                        hf['input'][idx, :, :, 0] = y_orig[y:y + height, x:x + width]
                        hf['input'][idx, :, :, 1] = qp
                        hf['label'][idx, :, :, 0] = diff[y:y + height, x:x + width]

                        idx += 1

                        hf['input'].resize((idx + 1, height, width, 2))
                        hf['label'].resize((idx + 1, height, width, 1))


def prepare_rate_data(data_name, input_dir, label_dir, width, height, levels, model, h5_dir):
    """
    Each line of a rate file includes:
    x y rate-level-1 rate-level-2 ... rate-level-n
    """
    file_name = '{0}_{1}_{2}x{3}.h5'.format(data_name, model, width, height)
    file_name = os.path.join(h5_dir, file_name)

    with h5py.File(file_name, 'w') as hf:
        hf.create_dataset('input', (1, height, width, 1), maxshape=(None, height, width, 1))
        hf.create_dataset('label', (1, levels), maxshape=(None, levels))

        orig_frames = sorted(glob(os.path.join(input_dir, '*.yuv')))
        rate_data = sorted(glob(os.path.join(label_dir, '*.txt')))

        idx = 0
        scale_factor = 2 ** 8 - 1.
        global_max = -1

        for i in range(len(orig_frames)):
            data_ = {}
            with open(rate_data[i], 'r') as file_:
                for line in file_:
                    line_ = np.fromstring(line, dtype=int, sep=' ')

                    rates = line_[2:]
                    rates = rates.astype('float')
                    data_['{0}x{1}'.format(line_[0], line_[1])] = rates
                    global_max = max(global_max, np.max(rates))

            w, h = shape_from_filename(orig_frames[i])
            y_orig = read_single_frame(orig_frames[i], w, h)
            w = width * (w // width)
            h = height * (h // height)
            y_orig = y_orig[:h, :w] / scale_factor

            for y in range(0, h, height):
                for x in range(0, w, width):
                    hf['input'][idx, :, :, 0] = y_orig[y:y + height, x:x + width]
                    hf['label'][idx, :] = data_['{0}x{1}'.format(x, y)]
                    idx += 1

                    hf['input'].resize((idx + 1, height, width, 1))
                    hf['label'].resize((idx + 1, levels))

        hf.attrs['global_max'] = global_max
