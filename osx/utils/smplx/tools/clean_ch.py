# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import os.path as osp

import pickle

from tqdm import tqdm
import numpy as np


def clean_fn(fn, output_folder='output'):
    with open(fn, 'rb') as body_file:
        body_data = pickle.load(body_file)

    output_dict = {}
    for key, data in body_data.iteritems():
        if 'chumpy' in str(type(data)):
            output_dict[key] = np.array(data)
        else:
            output_dict[key] = data

    out_fn = osp.split(fn)[1]

    out_path = osp.join(output_folder, out_fn)
    with open(out_path, 'wb') as out_file:
        pickle.dump(output_dict, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-models', dest='input_models', nargs='+',
                        required=True, type=str,
                        help='The path to the model that will be processed')
    parser.add_argument('--output-folder', dest='output_folder',
                        required=True, type=str,
                        help='The path to the output folder')

    args = parser.parse_args()

    input_models = args.input_models
    output_folder = args.output_folder
    if not osp.exists(output_folder):
        print('Creating directory: {}'.format(output_folder))
        os.makedirs(output_folder)

    for input_model in input_models:
        clean_fn(input_model, output_folder=output_folder)
