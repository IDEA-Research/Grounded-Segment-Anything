import numpy as np
import torch
import torch.nn as nn
from voxelnext_3d_box.utils import centernet_utils
import spconv.pytorch as spconv
import copy
from spconv.core import ConvAlgo


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, kernel_size, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size//2), bias=use_bias, indice_key=cur_name, algo=ConvAlgo.Native),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True, indice_key=cur_name+'out', algo=ConvAlgo.Native))
            fc = nn.Sequential(*fc_list)
            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x).features

        return ret_dict


class VoxelNeXtHead(nn.Module):
    def __init__(self, class_names, point_cloud_range, voxel_size, kernel_size_head,
                 CLASS_NAMES_EACH_HEAD, SEPARATE_HEAD_CFG, POST_PROCESSING):
        super().__init__()
        self.point_cloud_range = torch.Tensor(point_cloud_range)
        self.voxel_size = torch.Tensor(voxel_size)
        self.feature_map_stride = 8

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.POST_PROCESSING = POST_PROCESSING

        for cur_class_names in CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            ))
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=2)
            self.heads_list.append(
                SeparateHead(
                    input_channels=128,
                    sep_head_dict=cur_head_dict,
                    kernel_size=kernel_size_head,
                    use_bias=True,
                )
            )
        self.forward_ret_dict = {}

    def generate_predicted_boxes(self, batch_size, pred_dicts, voxel_indices, spatial_shape):
        device = pred_dicts[0]['hm'].device
        post_process_cfg = self.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).float().to(device)

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
            'pred_ious': [],
            'voxel_ids': []
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_iou = None
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
            voxel_indices_ = voxel_indices

            final_pred_dicts = centernet_utils.decode_bbox_from_voxels_nuscenes(
                batch_size=batch_size, indices=voxel_indices_,
                obj=batch_hm, 
                rot_cos=batch_rot_cos,
                rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z,
                dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range.to(device), voxel_size=self.voxel_size.to(device),
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range,
                add_features=torch.arange(voxel_indices_.shape[0], device=voxel_indices_.device).unsqueeze(-1)
            )

            for k, final_dict in enumerate(final_pred_dicts):
                class_id_mapping_each_head = self.class_id_mapping_each_head[idx].to(device)
                final_dict['pred_labels'] = class_id_mapping_each_head[final_dict['pred_labels'].long()]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])
                ret_dict[k]['pred_ious'].append(final_dict['pred_ious'])
                ret_dict[k]['voxel_ids'].append(final_dict['add_features'])

        for k in range(batch_size):
            pred_boxes = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            pred_scores = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            pred_labels = torch.cat(ret_dict[k]['pred_labels'], dim=0)
            voxel_ids = torch.cat(ret_dict[k]['voxel_ids'], dim=0)

            ret_dict[k]['pred_boxes'] = pred_boxes
            ret_dict[k]['pred_scores'] = pred_scores
            ret_dict[k]['pred_labels'] = pred_labels + 1
            ret_dict[k]['voxel_ids'] = voxel_ids

        return ret_dict

    def _get_voxel_infos(self, x):
        spatial_shape = x.spatial_shape
        voxel_indices = x.indices
        spatial_indices = []
        num_voxels = []
        batch_size = x.batch_size
        batch_index = voxel_indices[:, 0]

        for bs_idx in range(batch_size):
            batch_inds = batch_index==bs_idx
            spatial_indices.append(voxel_indices[batch_inds][:, [2, 1]])
            num_voxels.append(batch_inds.sum())

        return spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels

    def forward(self, data_dict):
        x = data_dict['encoded_spconv_tensor']
        spatial_shape, batch_index, voxel_indices, spatial_indices, num_voxels = self._get_voxel_infos(x)

        pred_dicts = []
        for idx, head in enumerate(self.heads_list):
            pred_dict = head(x)
            pred_dicts.append(pred_dict)

        pred_dicts = self.generate_predicted_boxes(
            data_dict['batch_size'],
            pred_dicts, voxel_indices, spatial_shape
        )

        return pred_dicts
