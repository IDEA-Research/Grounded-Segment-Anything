import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
from nets.resnet import ResNetBackbone
from nets.module_osx import PositionNet, FaceRegressor, BoxNet, BodyRotationNet, ScaleNet
from nets.module import PositionNet as PositionNet_H4W
from nets.module import RotationNet as RotationNet_H4W
from nets.module import HandRoI as HandRoI_H4W
from utils.human_models import smpl_x
from utils.transforms import rot6d_to_axis_angle, restore_bbox
from config import cfg
import math
import copy
from mmpose.models import build_posenet
from mmcv import Config

class Model(nn.Module):
    def __init__(self, backbone, body_position_net, body_rotation_net, box_net, hand_roi_net, hand_position_net, hand_rotation_net, face_regressor, scale_net):
        super(Model, self).__init__()
        # body
        self.backbone = backbone
        self.body_position_net = body_position_net
        self.body_rotation_net = body_rotation_net
        self.box_net = box_net

        # hand
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        self.hand_rotation_net = hand_rotation_net

        # face
        self.face_regressor = face_regressor

        self.scale_net = scale_net

        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()

        self.body_num_joints = len(smpl_x.pos_joint_part['body'])
        self.hand_joint_num = len(smpl_x.pos_joint_part['rhand'])

    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0] * cfg.focal[1] * cfg.camera_3d_size * cfg.camera_3d_size / (
                cfg.input_body_shape[0] * cfg.input_body_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)
        return cam_trans

    def get_coord(self, root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape, expr, cam_trans, mode):
        batch_size = root_pose.shape[0]
        zero_pose = torch.zeros((1, 3)).float().cuda().repeat(batch_size, 1)  # eye poses
        output = self.smplx_layer(betas=shape, body_pose=body_pose, global_orient=root_pose, right_hand_pose=rhand_pose,
                                  left_hand_pose=lhand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose,
                                  reye_pose=zero_pose, expression=expr)
        # camera-centered 3D coordinate
        vertices = output.vertices

        joint_cam = output.joints[:, smpl_x.joint_idx, :]

        # project 3D coordinates to 2D space
        x = (joint_cam[:, :, 0] + cam_trans[:, None, 0]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
            cfg.focal[0] + cfg.princpt[0]
        y = (joint_cam[:, :, 1] + cam_trans[:, None, 1]) / (joint_cam[:, :, 2] + cam_trans[:, None, 2] + 1e-4) * \
            cfg.focal[1] + cfg.princpt[1]
        x = x / cfg.input_body_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_body_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:, smpl_x.root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = vertices + cam_trans[:, None, :]  # for rendering

        # left hand root (left wrist)-relative 3D coordinatese
        lhand_idx = smpl_x.joint_part['lhand']
        lhand_cam = joint_cam[:, lhand_idx, :]
        lwrist_cam = joint_cam[:, smpl_x.lwrist_idx, None, :]
        lhand_cam = lhand_cam - lwrist_cam
        joint_cam = torch.cat((joint_cam[:, :lhand_idx[0], :], lhand_cam, joint_cam[:, lhand_idx[-1] + 1:, :]), 1)

        # right hand root (right wrist)-relative 3D coordinatese
        rhand_idx = smpl_x.joint_part['rhand']
        rhand_cam = joint_cam[:, rhand_idx, :]
        rwrist_cam = joint_cam[:, smpl_x.rwrist_idx, None, :]
        rhand_cam = rhand_cam - rwrist_cam
        joint_cam = torch.cat((joint_cam[:, :rhand_idx[0], :], rhand_cam, joint_cam[:, rhand_idx[-1] + 1:, :]), 1)

        # face root (neck)-relative 3D coordinates
        face_idx = smpl_x.joint_part['face']
        face_cam = joint_cam[:, face_idx, :]
        neck_cam = joint_cam[:, smpl_x.neck_idx, None, :]
        face_cam = face_cam - neck_cam
        joint_cam = torch.cat((joint_cam[:, :face_idx[0], :], face_cam, joint_cam[:, face_idx[-1] + 1:, :]), 1)

        return joint_proj, joint_cam, mesh_cam, vertices

    def forward(self, inputs, mode):

        # backbone
        body_img = F.interpolate(inputs['img'], cfg.input_body_shape)

        img_feat, task_tokens = self.backbone(body_img)  # task_token:[bs, N, c]

        shape_token, cam_token, expr_token, jaw_pose_token, hand_token, body_pose_token = \
            task_tokens[:, 0], task_tokens[:, 1], task_tokens[:, 2], task_tokens[:, 3], task_tokens[:, 4:6], task_tokens[:, 6:]

        # body
        body_joint_hm, body_joint_img = self.body_position_net(img_feat)

        # face
        expr, jaw_pose = self.face_regressor(expr_token, jaw_pose_token)
        jaw_pose = rot6d_to_axis_angle(jaw_pose)

        # hand bbox and feature extraction
        lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size, face_bbox_center, face_bbox_size = \
            self.box_net(img_feat, body_joint_hm.detach())
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0],
                                  2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0],
                                  2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space

        # hand
        hand_feat = self.hand_roi_net(inputs['img'], lhand_bbox,
                                      rhand_bbox)  # hand_feat: flipped left hand + right hand
        _, hand_joint_img = self.hand_position_net(hand_feat)  # (2N, J_P, 3)
        hand_pose = self.hand_rotation_net(hand_feat, hand_joint_img.detach())
        hand_pose = rot6d_to_axis_angle(hand_pose.reshape(-1, 6)).reshape(hand_feat.shape[0], -1)  # (2N, J_R*3)
        # restore flipped left hand joint coordinates
        batch_size = hand_joint_img.shape[0] // 2
        lhand_joint_img = hand_joint_img[:batch_size, :, :]
        lhand_joint_img = torch.cat(
            (cfg.output_hand_hm_shape[2] - 1 - lhand_joint_img[:, :, 0:1], lhand_joint_img[:, :, 1:]), 2)
        rhand_joint_img = hand_joint_img[batch_size:, :, :]
        # restore flipped left hand joint rotations
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size, :].reshape(-1, len(smpl_x.orig_joint_part['lhand']), 3)
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(batch_size, -1)
        rhand_pose = hand_pose[batch_size:, :]
        # restore flipped left hand features
        batch_size = hand_feat.shape[0] // 2
        lhand_feat = torch.flip(hand_feat[:batch_size, :], [3])
        rhand_feat = hand_feat[batch_size:, :]

        root_pose, body_pose, shape, cam_param = self.body_rotation_net(body_pose_token, shape_token, cam_token,
                                                                        body_joint_img.detach(), lhand_feat,
                                                                        lhand_joint_img[:,
                                                                        smpl_x.pos_joint_part['L_MCP'], :].detach(),
                                                                        rhand_feat,
                                                                        rhand_joint_img[:,
                                                                        smpl_x.pos_joint_part['L_MCP'], :].detach())
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(body_pose.shape[0], -1)  # (N, J_R*3)
        cam_trans = self.get_camera_trans(cam_param)

        # final output
        joint_proj, joint_cam, mesh_cam, vertices = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape,
                                                         expr, cam_trans, mode)

        # test output
        out = {}
        out['smplx_mesh_cam'] = mesh_cam
        return out


def init_weights(m):
    try:
        if type(m) == nn.ConvTranspose2d:
            nn.init.normal_(m.weight, std=0.001)
        elif type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    except AttributeError:
        pass

def get_model():

    # body
    VITPose_cfg = Config.fromfile(cfg.backbone_config_file)
    ViTPose = build_posenet(VITPose_cfg.model)
    body_position_net = PositionNet('body', feat_dim=cfg.feat_dim)
    body_rotation_net = BodyRotationNet(feat_dim=cfg.feat_dim)
    box_net = BoxNet(feat_dim=cfg.feat_dim)

    # hand
    hand_backbone = ResNetBackbone(cfg.hand_resnet_type)
    hand_roi_net = HandRoI_H4W(hand_backbone)
    hand_position_net = PositionNet_H4W('hand', cfg.hand_resnet_type)
    hand_rotation_net = RotationNet_H4W('hand', cfg.hand_resnet_type)

    # face
    face_regressor = FaceRegressor(feat_dim=cfg.feat_dim)

    # scale
    scale_net = ScaleNet()

    backbone = ViTPose.backbone
    model = Model(backbone, body_position_net, body_rotation_net, box_net, hand_roi_net, hand_position_net,
                  hand_rotation_net, face_regressor, scale_net)

    return model
