import sys

import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import PositionNet, HandRotationNet, FaceRegressor, BoxNet, HandRoI, BodyRotationNet

from utils.human_models import smpl_x
from utils.transforms import rot6d_to_axis_angle, restore_bbox
from config import cfg
import math
import copy
from mmpose.models import build_posenet
from mmcv import Config

class Model(nn.Module):
    def __init__(self, encoder, body_position_net, body_rotation_net, box_net, hand_position_net, hand_roi_net,
                 hand_rotation_net, face_regressor):
        super(Model, self).__init__()

        # body
        self.backbone = encoder
        self.body_position_net = body_position_net
        self.body_rotation_net = body_rotation_net
        self.box_net = box_net

        # hand
        self.hand_roi_net = hand_roi_net
        self.hand_position_net = hand_position_net
        self.hand_rotation_net = hand_rotation_net

        # face
        self.face_regressor = face_regressor

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
        # root-relative 3D coordinates
        mesh_cam = vertices + cam_trans[:, None, :]  # for rendering

        return mesh_cam

    def forward(self, inputs, mode):

        # backbone
        body_img = F.interpolate(inputs['img'], cfg.input_body_shape)

        # 1. Encoder
        img_feat, task_tokens = self.backbone(body_img)  # task_token:[bs, N, c]
        shape_token, cam_token, expr_token, jaw_pose_token, hand_token, body_pose_token = \
            task_tokens[:, 0], task_tokens[:, 1], task_tokens[:, 2], task_tokens[:, 3], task_tokens[:,
                                                                                        4:6], task_tokens[:, 6:]

        # 2. Body Regressor
        body_joint_hm, body_joint_img = self.body_position_net(img_feat)
        root_pose, body_pose, shape, cam_param, = self.body_rotation_net(body_pose_token, shape_token, cam_token,
                                                                      body_joint_img.detach())
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(body_pose.shape[0], -1)  # (N, J_R*3)
        cam_trans = self.get_camera_trans(cam_param)

        # 3. Hand and Face BBox Estimation
        lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size, face_bbox_center, face_bbox_size = self.box_net(
            img_feat, body_joint_hm.detach())
        lhand_bbox = restore_bbox(lhand_bbox_center, lhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0],
                                  2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        rhand_bbox = restore_bbox(rhand_bbox_center, rhand_bbox_size, cfg.input_hand_shape[1] / cfg.input_hand_shape[0],
                                  2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space

        # 4. Differentiable Feature-level Hand Crop-Upsample
        # hand_feat: list, [bsx2, c, cfg.output_hm_shape[1]*scale, cfg.output_hm_shape[2]*scale]
        hand_feat = self.hand_roi_net(img_feat, lhand_bbox, rhand_bbox)  # hand_feat: flipped left hand + right hand

        # 5. Hand/Face Regressor
        # hand regressor
        _, hand_joint_img = self.hand_position_net(hand_feat)  # (2N, J_P, 3)
        hand_pose = self.hand_rotation_net(hand_feat, hand_joint_img.detach())
        hand_pose = rot6d_to_axis_angle(hand_pose.reshape(-1, 6)).reshape(hand_feat.shape[0], -1)  # (2N, J_R*3)
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size, :].reshape(-1, len(smpl_x.orig_joint_part['lhand']), 3)
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(batch_size, -1)
        rhand_pose = hand_pose[batch_size:, :]

        # hand regressor
        expr, jaw_pose = self.face_regressor(expr_token, jaw_pose_token)
        jaw_pose = rot6d_to_axis_angle(jaw_pose)

        # final output
        mesh_cam = self.get_coord(root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, shape,
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
    vit_cfg = Config.fromfile(cfg.encoder_config_file)
    vit = build_posenet(vit_cfg.model)
    body_position_net = PositionNet('body', feat_dim=cfg.feat_dim)
    body_rotation_net = BodyRotationNet(feat_dim=cfg.feat_dim)
    box_net = BoxNet(feat_dim=cfg.feat_dim)

    # hand
    hand_position_net = PositionNet('hand', feat_dim=cfg.feat_dim)
    hand_roi_net = HandRoI(feat_dim=cfg.feat_dim, upscale=cfg.upscale)
    hand_rotation_net = HandRotationNet('hand', feat_dim=cfg.feat_dim)

    # face
    face_regressor = FaceRegressor(feat_dim=cfg.feat_dim)

    # scale
    encoder = vit.backbone
    model = Model(encoder, body_position_net, body_rotation_net, box_net, hand_position_net, hand_roi_net,
                  hand_rotation_net,
                  face_regressor)

    return model
