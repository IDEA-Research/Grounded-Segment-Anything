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

import os
import os.path as osp
import pickle

import argparse

import numpy as np


def merge_models(smplh_fn, mano_left_fn, mano_right_fn,
                 output_folder='output'):

    with open(smplh_fn, 'rb') as body_file:
        body_data = pickle.load(body_file)

    with open(mano_left_fn, 'rb') as lhand_file:
        lhand_data = pickle.load(lhand_file)

    with open(mano_right_fn, 'rb') as rhand_file:
        rhand_data = pickle.load(rhand_file)

    out_fn = osp.split(smplh_fn)[1]

    output_data = body_data.copy()
    output_data['hands_componentsl'] = lhand_data['hands_components']
    output_data['hands_componentsr'] = rhand_data['hands_components']

    output_data['hands_coeffsl'] = lhand_data['hands_coeffs']
    output_data['hands_coeffsr'] = rhand_data['hands_coeffs']

    output_data['hands_meanl'] = lhand_data['hands_mean']
    output_data['hands_meanr'] = rhand_data['hands_mean']

    for key, data in output_data.iteritems():
        if 'chumpy' in str(type(data)):
            output_data[key] = np.array(data)
        else:
            output_data[key] = data

    out_path = osp.join(output_folder, out_fn)
    print(out_path)
    print('Saving to {}'.format(out_path))
    with open(out_path, 'wb') as output_file:
        pickle.dump(output_data, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplh-fn', dest='smplh_fn', required=True,
                        type=str, help='The path to the SMPLH model')
    parser.add_argument('--mano-left-fn', dest='mano_left_fn', required=True,
                        type=str, help='The path to the left hand MANO model')
    parser.add_argument('--mano-right-fn', dest='mano_right_fn', required=True,
                        type=str, help='The path to the right hand MANO model')
    parser.add_argument('--output-folder', dest='output_folder',
                        required=True, type=str,
                        help='The path to the output folder')

    args = parser.parse_args()

    smplh_fn = args.smplh_fn
    mano_left_fn = args.mano_left_fn
    mano_right_fn = args.mano_right_fn
    output_folder = args.output_folder

    if not osp.exists(output_folder):
        print('Creating directory: {}'.format(output_folder))
        os.makedirs(output_folder)

    merge_models(smplh_fn, mano_left_fn, mano_right_fn, output_folder)
