# This file is modified from https://github.com/tianweiy/CenterPoint

import torch


def _topk_1d(scores, batch_size, batch_idx, obj, K=40, nuscenes=False):
    # scores: (N, num_classes)
    topk_score_list = []
    topk_inds_list = []
    topk_classes_list = []

    for bs_idx in range(batch_size):
        batch_inds = batch_idx==bs_idx
        if obj.shape[-1] == 1 and not nuscenes:
            score = scores[batch_inds].permute(1, 0)
            topk_scores, topk_inds = torch.topk(score, K)
            topk_score, topk_ind = torch.topk(obj[topk_inds.view(-1)].squeeze(-1), K) #torch.topk(topk_scores.view(-1), K)
        else:
            score = obj[batch_inds].permute(1, 0)
            topk_scores, topk_inds = torch.topk(score, min(K, score.shape[-1]))
            topk_score, topk_ind = torch.topk(topk_scores.view(-1), min(K, topk_scores.view(-1).shape[-1]))

        topk_classes = (topk_ind // K).int()
        topk_inds = topk_inds.view(-1).gather(0, topk_ind)
        #print('topk_inds', topk_inds)

        if not obj is None and obj.shape[-1] == 1:
            topk_score_list.append(obj[batch_inds][topk_inds])
        else:
            topk_score_list.append(topk_score)
        topk_inds_list.append(topk_inds)
        topk_classes_list.append(topk_classes)

    topk_score = torch.stack(topk_score_list)
    topk_inds = torch.stack(topk_inds_list)
    topk_classes = torch.stack(topk_classes_list)

    return topk_score, topk_inds, topk_classes

def gather_feat_idx(feats, inds, batch_size, batch_idx):
    feats_list = []
    dim = feats.size(-1)
    _inds = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), dim)

    for bs_idx in range(batch_size):
        batch_inds = batch_idx==bs_idx
        feat = feats[batch_inds]
        feats_list.append(feat.gather(0, _inds[bs_idx]))
    feats = torch.stack(feats_list)
    return feats


def decode_bbox_from_voxels_nuscenes(batch_size, indices, obj, rot_cos, rot_sin,
                            center, center_z, dim, vel=None, iou=None, point_cloud_range=None, voxel_size=None, voxels_3d=None,
                            feature_map_stride=None, K=100, score_thresh=None, post_center_limit_range=None, add_features=None):
    batch_idx = indices[:, 0]
    spatial_indices = indices[:, 1:]
    scores, inds, class_ids = _topk_1d(None, batch_size, batch_idx, obj, K=K, nuscenes=True)

    center = gather_feat_idx(center, inds, batch_size, batch_idx)
    rot_sin = gather_feat_idx(rot_sin, inds, batch_size, batch_idx)
    rot_cos = gather_feat_idx(rot_cos, inds, batch_size, batch_idx)
    center_z = gather_feat_idx(center_z, inds, batch_size, batch_idx)
    dim = gather_feat_idx(dim, inds, batch_size, batch_idx)
    spatial_indices = gather_feat_idx(spatial_indices, inds, batch_size, batch_idx)

    if not add_features is None:
        add_features = gather_feat_idx(add_features, inds, batch_size, batch_idx) #for add_feature in add_features]

    if not isinstance(feature_map_stride, int):
        feature_map_stride = gather_feat_idx(feature_map_stride.unsqueeze(-1), inds, batch_size, batch_idx)

    angle = torch.atan2(rot_sin, rot_cos)
    xs = (spatial_indices[:, :, -1:] + center[:, :, 0:1]) * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = (spatial_indices[:, :, -2:-1] + center[:, :, 1:2]) * feature_map_stride * voxel_size[1] + point_cloud_range[1]

    box_part_list = [xs, ys, center_z, dim, angle]

    if not vel is None:
        vel = gather_feat_idx(vel, inds, batch_size, batch_idx)
        box_part_list.append(vel)

    if not iou is None:
        iou = gather_feat_idx(iou, inds, batch_size, batch_idx)
        iou = torch.clamp(iou, min=0, max=1.)

    final_box_preds = torch.cat((box_part_list), dim=-1)
    final_scores = scores.view(batch_size, K)
    final_class_ids = class_ids.view(batch_size, K)
    if not add_features is None:
        add_features = add_features.view(batch_size, K, add_features.shape[-1]) #for add_feature in add_features]

    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

    if score_thresh is not None:
        mask &= (final_scores > score_thresh)

    ret_pred_dicts = []
    for k in range(batch_size):
        cur_mask = mask[k]
        cur_boxes = final_box_preds[k, cur_mask]
        cur_scores = final_scores[k, cur_mask]
        cur_labels = final_class_ids[k, cur_mask]
        cur_add_features = add_features[k, cur_mask] if not add_features is None else None
        cur_iou = iou[k, cur_mask] if not iou is None else None

        ret_pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels,
            'pred_ious': cur_iou,
            'add_features': cur_add_features,
        })
    return ret_pred_dicts
