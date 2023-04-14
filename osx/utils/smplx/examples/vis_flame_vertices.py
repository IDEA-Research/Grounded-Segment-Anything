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
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import argparse
import pickle

import numpy as np
import torch
import open3d as o3d

import smplx


def main(model_folder, corr_fname, ext='npz',
         head_color=(0.3, 0.3, 0.6),
         gender='neutral'):

    head_idxs = np.load(corr_fname)

    model = smplx.create(model_folder, model_type='smplx',
                         gender=gender,
                         ext=ext)
    betas = torch.zeros([1, 10], dtype=torch.float32)
    expression = torch.zeros([1, 10], dtype=torch.float32)

    output = model(betas=betas, expression=expression,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(model.faces)
    mesh.compute_vertex_normals()

    colors = np.ones_like(vertices) * [0.3, 0.3, 0.3]
    colors[head_idxs] = head_color

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--corr-fname', required=True, type=str,
                        dest='corr_fname',
                        help='Filename with the head correspondences')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--head', default='right',
                        choices=['right', 'left'],
                        type=str, help='Which head to plot')
    parser.add_argument('--head-color', type=float, nargs=3, dest='head_color',
                        default=(0.3, 0.3, 0.6),
                        help='Color for the head vertices')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    corr_fname = args.corr_fname
    gender = args.gender
    ext = args.ext
    head = args.head
    head_color = args.head_color

    main(model_folder, corr_fname, ext=ext,
         head_color=head_color,
         gender=gender
         )
